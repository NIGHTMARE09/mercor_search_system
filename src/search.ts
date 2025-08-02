import { MongoClient, Db, ObjectId } from 'mongodb';
// import OpenAI from 'openai';
import Turbopuffer from '@turbopuffer/turbopuffer';
import dotenv from 'dotenv';
import { VoyageAIClient, VoyageAIError } from "voyageai";
dotenv.config();



// const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const voyage = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY || '' });
const tpuffer = new Turbopuffer({
    apiKey: process.env.TURBOPUFFER_API_KEY,
    baseURL: process.env.TURBOPUFFER_BASE_URL || 'https://us-east-1.turbopuffer.com',
});
const mongoUri = process.env.MONGODB_URI || '';
const client = new MongoClient(mongoUri, {
    serverSelectionTimeoutMS: 5000,
    retryWrites: true,
    w: 'majority',
    maxPoolSize: 10,
});
let db: Db;



async function connectDb() {
    await client.connect();
    db = client.db('interview_data');
    console.log('Connected to MongoDB');
}



interface Profile {
    _id: ObjectId;
    name: string;
    country: string;
    skills: string[];
    yearsOfWorkExperience: number;
    embedding: number[];
    rerankSummary: string;
    awardsAndCertifications: { name: string }[];
}



interface Query {
    text: string;
    rerankSummary?: string;
    hardCriteria: {
        skills?: string[];
        experience?: number | { min: number; max?: number };  // Now supports number or range object
    };
    softCriteria: {
        skills?: string[];
        country?: string;
        diversity?: boolean;
    };
}

// Generate embedding using VoyageAI
async function generateEmbedding(text: string): Promise<number[]> {
    try {
        const response = await voyage.embed({
            model: "voyage-3",
            input: text || "default query",
        });
        if (response.data && response.data.length > 0 && response.data[0].embedding) {
            return response.data[0].embedding;
        } else {
            console.error("Embedding API returned no data.");
            return new Array(1024).fill(0); // Fallback vector if API returns no data
        }
    } catch (error) {
        console.error("Embedding error:", error);
        if (error instanceof VoyageAIError) {
            console.log(error.statusCode);
            console.log(error.message);
            console.log(error.body);
        }
        return new Array(1024).fill(0); // Fallback vector if API fails
    }
}



// Cosine Similarity for Vector Matching
function calculateCosineSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length || vec1.length === 0) return 0;
    const dotProduct = vec1.reduce((sum, a, i) => sum + a * vec2[i], 0);
    const magnitude1 = Math.sqrt(vec1.reduce((sum, a) => sum + a * a, 0));
    const magnitude2 = Math.sqrt(vec2.reduce((sum, a) => sum + a * a, 0));
    return magnitude1 * magnitude2 === 0 ? 0 : dotProduct / (magnitude1 * magnitude2);
}



// Helper function to filter valid ObjectId strings
function isValidObjectId(id: string): boolean {
    return /^[0-9a-fA-F]{24}$/.test(id);  // Check for 24 hex chars
}



// Helper function to check if a profile passes hard criteria (returns 1 or 0)
function passesHardCriteria(p: Profile, h: Query['hardCriteria']): number {
    let hardMatch = 1;
    if (h.skills && h.skills.length > 0) {
        if (!h.skills.every(s => {
            const hardKeywords = s.toLowerCase().split(' ');
            return hardKeywords.some(kw => p.skills.some(ps => ps.toLowerCase().includes(kw.toLowerCase())) || p.rerankSummary.toLowerCase().includes(kw.toLowerCase()));
        })) {
            hardMatch = 0;
        }
    }
    if (h.experience) {
        let minExp = typeof h.experience === 'number' ? h.experience : h.experience.min;
        let maxExp = typeof h.experience === 'number' ? undefined : h.experience.max;
        if (p.yearsOfWorkExperience < minExp || (maxExp !== undefined && p.yearsOfWorkExperience > maxExp)) {
            hardMatch = 0;
        }
    }
    return hardMatch;
}



// Updated searchProfiles
async function searchProfiles(query: Query): Promise<{ id: string; score: number }[]> {
    console.log('Search query:', query);
    if (!query) {
        throw new Error('Query parameter is undefined');
    }
    if (!query.text) {
        throw new Error('Query text is missing');
    }
    if (query.hardCriteria.experience && typeof query.hardCriteria.experience === 'string') {
        const expStr: string = query.hardCriteria.experience;
        const rangeMatch = expStr.match(/(\d+)-(\d+)/);
        if (rangeMatch) {
            query.hardCriteria.experience = { min: parseInt(rangeMatch[1]), max: parseInt(rangeMatch[2]) };
        } else {
            query.hardCriteria.experience = { min: parseInt(expStr) };
        }
    }

    await connectDb();


    const queryEmbedding = await generateEmbedding(query.text + (query.rerankSummary ? ` ${query.rerankSummary}` : ""));


    const ns = tpuffer.namespace('mercor-profiles');
        const vectorResults = await ns.query({
        rank_by: ['vector', 'ANN', queryEmbedding],
        top_k: 200,
    });
    

    const rows = vectorResults.rows ?? [];
    const candidateIds = rows.map(r => r.id.toString());


    const validCandidateIds = candidateIds.filter(id => isValidObjectId(id));


    console.log(`Total candidates from TurboPuffer: ${candidateIds.length}, Valid IDs: ${validCandidateIds.length}`);


    if (validCandidateIds.length === 0) {
        return [];
    }


    const profiles = await db.collection<Profile>('linkedin_data_subset').find({
        _id: { $in: validCandidateIds.map(id => new ObjectId(id)) }
    }).toArray();


    const hardFiltered = profiles.filter(p => passesHardCriteria(p, query.hardCriteria) === 1);


    const scored = hardFiltered.map(p => {
        // Recompute hardMatch here using helper function
        const hardMatch = passesHardCriteria(p, query.hardCriteria);

        let softAvg = 0;
        const soft = query.softCriteria;
        let softCount = 0;


        if (soft.skills && soft.skills.length > 0) {
            const matched = soft.skills.filter(s => {
                const softKeywords = s.toLowerCase().split(' ');
                return softKeywords.some(kw => p.skills.some(ps => ps.toLowerCase().includes(kw.toLowerCase())) || p.rerankSummary.toLowerCase().includes(kw.toLowerCase()));
            }).length;
            softAvg += matched / soft.skills.length;
            softCount++;
        }


        if (soft.country) {
            softAvg += (p.country && p.country.toLowerCase() === soft.country.toLowerCase()) ? 1 : 0;
            softCount++;
        }


        if (soft.diversity) {
            softAvg += p.awardsAndCertifications.some(a => a.name.toLowerCase().includes('diversity')) ? 1 : 0;
            softCount++;
        }


        softAvg = softCount > 0 ? softAvg / softCount : 0;


        const vectorSim = calculateCosineSimilarity(queryEmbedding, p.embedding || new Array(1024).fill(0.001));


        const finalScore = hardMatch * softAvg * vectorSim;


        console.log(`Candidate ID: ${p._id.toString()}, softAvg: ${softAvg}, vectorSim: ${vectorSim}, finalScore: ${finalScore}`);  // Debug log


        return { id: p._id.toString(), score: finalScore };
    });


    scored.sort((a, b) => b.score - a.score);
    let results = scored.slice(0, 10);


    if (results.length < 10) {
        console.log(`Padding results for query - only ${results.length} hard matches found. Config: ${JSON.stringify(query)}`);
        const nonHardScored = profiles.filter(p => passesHardCriteria(p, query.hardCriteria) === 0).map(p => {
            let softAvg = 0;
            const soft = query.softCriteria;
            let softCount = 0;


            if (soft.skills && soft.skills.length > 0) {
                const matched = soft.skills.filter(s => {
                    const softKeywords = s.toLowerCase().split(' ');
                    return softKeywords.some(kw => p.skills.some(ps => ps.toLowerCase().includes(kw.toLowerCase())) || p.rerankSummary.toLowerCase().includes(kw.toLowerCase()));
                }).length;
                softAvg += matched / soft.skills.length;
                softCount++;
            }


            if (soft.country) {
                softAvg += (p.country && p.country.toLowerCase() === soft.country.toLowerCase()) ? 1 : 0;
                softCount++;
            }


            if (soft.diversity) {
                softAvg += p.awardsAndCertifications.some(a => a.name.toLowerCase().includes('diversity')) ? 1 : 0;
                softCount++;
            }


            softAvg = softCount > 0 ? softAvg / softCount : 0;


            const vectorSim = calculateCosineSimilarity(queryEmbedding, p.embedding || new Array(1024).fill(0.001));


            return { id: p._id.toString(), score: softAvg * vectorSim };
        });


        nonHardScored.sort((a, b) => b.score - a.score);
        results = results.concat(nonHardScored.slice(0, 10 - results.length));
    }


    return results;
}



async function upsertProfiles() {
    await connectDb();
    console.log("Starting upsert in batches...");


    // Delete existing namespace to clear invalid IDs (using client-level method)
    try {
        await tpuffer.delete<any>('mercor-profiles');  // Uses tpuffer.delete() to remove a namespace. See: https://docs.turbopuffer.com/reference/delete-namespace
        console.log("Old namespace deleted successfully.");
    } catch (err) {
        console.error("Error deleting namespace (might not exist or permission issue):", err);
        // If delete fails, still continue upsert (it will overwrite)
    }


    const batchSize = 1000;
    let skip = 0;
    let hasMore = true;


    const ns = tpuffer.namespace('mercor-profiles');


    while (hasMore) {
        const profiles = await db.collection<Profile>('linkedin_data_subset')
            .find()
            .skip(skip)
            .limit(batchSize)
            .toArray();


        if (profiles.length === 0) {
            hasMore = false;
            break;
        }


        console.log(`Upserting batch of ${profiles.length} profiles...`);


        try {
            await ns.write({
                distance_metric: 'cosine_distance',
                upsert_rows: profiles.map((p, index) => {
                    let id = p._id ? p._id.toString() : new ObjectId().toString();  // Always valid 24-hex ID
                    console.log(`Profile ${index + skip}: ID=${id}, Name=${p.name || "Unknown"}`);
                    return {
                        id: id,
                        vector: p.embedding && Array.isArray(p.embedding) ? p.embedding : new Array(1024).fill(0)
                    };
                })
            });
            console.log(`Successfully upserted batch of ${profiles.length} profiles at skip=${skip}.`);
        } catch (err) {
            console.error(`Error upserting batch at skip=${skip}:`, err);
        }


        skip += batchSize;
    }


    console.log("All batches upserted successfully!");
}



export { searchProfiles, upsertProfiles, generateEmbedding };

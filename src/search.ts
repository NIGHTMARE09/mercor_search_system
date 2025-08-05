import { MongoClient, Db, ObjectId } from 'mongodb';
import Turbopuffer from '@turbopuffer/turbopuffer';
import dotenv from 'dotenv';
import { VoyageAIClient, VoyageAIError } from "voyageai";
import FlexSearch, { Index as FlexSearchIndex } from 'flexsearch';
dotenv.config();

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
        experience?: string;
    };
    softCriteria: {
        skills?: string[];
        country?: string;
        diversity?: boolean;
    };
}

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
            return new Array(1024).fill(0);
        }
    } catch (error) {
        console.error("Embedding error:", error);
        return new Array(1024).fill(0);
    }
}

function calculateCosineSimilarity(vec1: number[], vec2: number[]): number {
    if (vec1.length !== vec2.length || vec1.length === 0) return 0;
    const dotProduct = vec1.reduce((sum, a, i) => sum + a * vec2[i], 0);
    const magnitude1 = Math.sqrt(vec1.reduce((sum, a) => sum + a * a, 0));
    const magnitude2 = Math.sqrt(vec2.reduce((sum, a) => sum + a * a, 0));
    return magnitude1 * magnitude2 === 0 ? 0 : dotProduct / (magnitude1 * magnitude2);
}

function isValidObjectId(id: string): boolean {
    return /^[0-9a-fA-F]{24}$/.test(id);
}

function extractMeaningfulKeywords(sentence: string): string[] {
    const stopWords = ['with', 'and', 'or', 'such', 'as', 'in', 'of', 'the', 'a', 'an', 'experience', 'familiarity', 'research', 'completed', 'from', 'top', 'us', 'university', 'phd', 'mba', 'two', 'plus', 'years', '1.', '2.', '3.'];
    
    const parts = sentence.toLowerCase()
        .replace(/[1-3]\./g, '')
        .replace(/[+]/g, '')
        .split(/[,;]/)
        .join(' ')
        .split(/\s+/)
        .filter(word => word.length > 2 && !stopWords.includes(word));
    
    const keywords = [];
    for (let i = 0; i < parts.length; i++) {
        keywords.push(parts[i]);
        if (i < parts.length - 1) keywords.push(`${parts[i]} ${parts[i + 1]}`);
        if (i < parts.length - 2) keywords.push(`${parts[i]} ${parts[i + 1]} ${parts[i + 2]}`);
    }
    
    return [...new Set(keywords)];
}

function passesHardCriteria(p: Profile, h: Query['hardCriteria'], flexIndex: FlexSearchIndex): number {
    let hardScore = 0;
    let criteriaCount = 0;

    if (h.skills && h.skills.length > 0) {
        criteriaCount += h.skills.length;
        let matched = 0;
        h.skills.forEach(s => {
            const keywords = extractMeaningfulKeywords(s);
            const numMatched = keywords.filter(kw => flexIndex.search(kw).length > 0).length;
            matched += numMatched / keywords.length;
        });
        hardScore += matched / h.skills.length;
    }

    if (h.experience) {
        criteriaCount += 1;
        const expStr = h.experience.replace('+', '');
        const minExp = parseInt(expStr) || 0;
        const yoe = p.yearsOfWorkExperience;
        if (yoe >= minExp) {
            hardScore += 1;
        } else if (yoe >= minExp - 2) {
            hardScore += (yoe / minExp);
        }
    }

    return criteriaCount > 0 ? hardScore / criteriaCount : 0;
}

async function searchProfiles(query: Query): Promise<{ id: string; score: number }[]> {
    console.log('Search query:', query);
    if (!query) throw new Error('Query parameter is undefined');
    if (!query.text) throw new Error('Query text is missing');

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

    if (validCandidateIds.length === 0) return [];

    const profiles = await db.collection<Profile>('linkedin_data_subset').find({
        _id: { $in: validCandidateIds.map(id => new ObjectId(id)) }
    }).toArray();

    // FlexSearch with synonym map (fixed!)
    const synonymMap = {
        phd: ['doctorate', 'ph.d', 'doctoral'],
        gaussian: ['qm software', 'quantum tool', 'quantum chemistry software', 'vasp', 'pyscf'],
        putnam: ['math contest', 'putnam competition', 'math competition'],
        imo: ['international math olympiad', 'math olympiad'],
        usamo: ['usa math olympiad', 'math olympiad'],
        coo: ['chief operating officer', 'operations lead', 'vp operations'],
        jd: ['juris doctor', 'law degree'],
        mpp: ['master public policy', 'public policy degree'],
        ml: ['machine learning', 'ai', 'deep learning'],
        // Add more for other configs
    };

    const flexIndex = new FlexSearchIndex({
        tokenize: 'forward',
        cache: true,
        context: true,
        ...(synonymMap as any) // Pass synonym map here
    });

    profiles.forEach(p => {
        let content = [p.skills.join(' '), p.rerankSummary, p.country, p.awardsAndCertifications.map(a => a.name).join(' '), p.name].join(' ').toLowerCase();
        flexIndex.add(p._id.toString(), content);
    });

    const scored = profiles.map(p => {
        const hardScore = passesHardCriteria(p, query.hardCriteria, flexIndex);

        let softAvg = 0;
        const soft = query.softCriteria;
        let softCount = 0;

        if (soft.skills && soft.skills.length > 0) {
            let totalMatched = 0;
            soft.skills.forEach(skillSentence => {
                const keywords = extractMeaningfulKeywords(skillSentence);
                const matched = keywords.filter(kw => flexIndex.search(kw).length > 0).length;
                totalMatched += matched / keywords.length;
            });
            softAvg += totalMatched / soft.skills.length;
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

        const finalScore = (hardScore * 0.6) + (softAvg * 0.3) + (vectorSim * 0.1);

        console.log(`Candidate ID: ${p._id.toString()}, hardScore: ${hardScore}, softAvg: ${softAvg}, vectorSim: ${vectorSim}, finalScore: ${finalScore}`);

        return { id: p._id.toString(), score: finalScore };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 10);
}



async function upsertProfiles() {
    await connectDb();
    console.log("Starting upsert in batches...");

    try {
        await tpuffer.delete<any>('mercor-profiles');
        console.log("Old namespace deleted successfully.");
    } catch (err) {
        console.error("Error deleting namespace (might not exist or permission issue):", err);
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
                    let id = p._id ? p._id.toString() : new ObjectId().toString();
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

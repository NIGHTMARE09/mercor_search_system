import express from 'express';
import axios from 'axios';
import {searchProfiles, upsertProfiles} from './search';
const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));
app.post('/upsert', async(req, res) => {
    try{
        await upsertProfiles();
        console.log('Upsert request received and completed successfully');
        res.status(200).json({message: 'Profiles upserted successfully'});
    }
    catch(error) {
        console.error('Error upserting profiles:', error);
        res.status(500).json({error: 'Failed to upsert profiles'});
    }
});

app.post('/search', async(req, res) =>{
    try{
        console.log('Request Body:', req.body);
        const query = req.body;
        console.log('Search Query:', query);
        if(!query || !query.text){
            console.log('Validation failed:', { query, hasText: !!query?.text });
            return res.status(400).json({error: 'Invalid query format - text field is required'});
        }
        console.log('Calling searchProfiles with query:', query); 
        const results = await searchProfiles(query);
        console.log('Search results:', results);
        res.status(200).json(results);
    }
    catch(error: any){
        console.error('Error searching profiles:', error);
        res.status(500).json({error: 'Failed to search profiles', details: error.message});
    }
})

app.post('/grade', async(req, res) =>{
    
    const {config_candidates} = req.body;
    try{
        const response = await axios.post(process.env.GRADING_ENDPOINT || '', config_candidates,  {
            headers: {
                'Authorization' : process.env.YOUR_EMAIL || '',
                'Content-Type': 'application/json',
            },
        });
        res.status(200).json(response.data);
    }
    catch(error){
        console.error('Error grading profiles:', error);
        res.status(500).json({error: 'Failed to grade profiles'});
    }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
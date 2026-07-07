import * as path from 'path';
import { fileURLToPath } from 'url';
import { seedCompanies, closeConnection } from '../src/index';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const csvPath = path.resolve(__dirname, '../../websummit_lisbon2025_companies.csv');

async function run() {
  const uri = process.env.MONGODB_URI || 'mongodb://localhost:27017';
  const dbName = process.env.MONGODB_DB_NAME || 'distance_metrics';
  
  console.log('Starting onboarding to MongoDB...');
  console.log(`Connecting to URI: ${uri.replace(/\/\/.*@/, '//****:****@')}`); // Hide credentials
  console.log(`Using Database: ${dbName}`);
  
  try {
    const count = await seedCompanies(csvPath, {
      uri,
      dbName,
      preferOpenAI: !!process.env.OPENAI_API_KEY,
    });
    console.log(`Successfully onboarded and seeded ${count} companies into MongoDB!`);
  } catch (error) {
    console.error('Error onboarding dataset:', error);
  } finally {
    await closeConnection();
  }
}

run();

import neo4j from 'neo4j-driver';

let driver;

export function getDriver() {
  if (!driver) {
    driver = neo4j.driver(
      process.env.NEO4J_URI,
      neo4j.auth.basic(process.env.NEO4J_USERNAME, process.env.NEO4J_PASSWORD)
    );
  }
  return driver;
}

export async function runQuery(cypher, params = {}) {
  const session = getDriver().session();
  try {
    const result = await session.run(cypher, params);
    return result.records.map(r => r.toObject());
  } finally {
    await session.close();
  }
}
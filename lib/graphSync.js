import { runQuery } from './neo4j';

/**
 * Call this after a transaction is created/updated in Prisma.
 * Builds: (User)-[:SPENT_ON]->(Category) and links categories
 * that co-occur for the same user (spending pattern graph).
 */
export async function syncTransactionToGraph(transaction) {
  const { userId, category, amount, date, merchant, type } = transaction;

  // Only track expenses for spending pattern graph
  if (type !== 'EXPENSE') return;

  await runQuery(
    `
    MERGE (u:User {id: $userId})
    MERGE (c:Category {name: $category})
    MERGE (u)-[r:SPENT_ON]->(c)
    ON CREATE SET r.totalAmount = $amount, r.count = 1
    ON MATCH SET r.totalAmount = r.totalAmount + $amount, r.count = r.count + 1
    SET r.lastDate = $date
    `,
    { userId, category, amount, date: date.toISOString() }
  );

  if (merchant) {
    await runQuery(
      `
      MERGE (c:Category {name: $category})
      MERGE (m:Merchant {name: $merchant})
      MERGE (c)-[r:AT_MERCHANT]->(m)
      ON CREATE SET r.count = 1
      ON MATCH SET r.count = r.count + 1
      `,
      { category, merchant }
    );
  }
}

/**
 * Builds CO_OCCURS_WITH relationships between categories
 * a user spends on within the same month — for "users who spend on X also spend on Y".
 */
export async function buildCategoryCoOccurrence(userId) {
  await runQuery(
    `
    MATCH (u:User {id: $userId})-[:SPENT_ON]->(c1:Category)
    MATCH (u)-[:SPENT_ON]->(c2:Category)
    WHERE c1.name < c2.name
    MERGE (c1)-[r:CO_OCCURS_WITH]-(c2)
    ON CREATE SET r.weight = 1
    ON MATCH SET r.weight = r.weight + 1
    `,
    { userId }
  );
}
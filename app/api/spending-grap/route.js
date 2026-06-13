import { auth } from '@clerk/nextjs/server';
import { runQuery } from '@/lib/neo4j';
import { NextResponse } from 'next/server';

export async function GET() {
  const { userId } = await auth();
  if (!userId) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const spending = await runQuery(
    `
    MATCH (u:User {id: $userId})-[r:SPENT_ON]->(c:Category)
    RETURN c.name AS category, r.totalAmount AS total, r.count AS count
    ORDER BY r.totalAmount DESC
    `,
    { userId }
  );

  const coOccurrence = await runQuery(
    `
    MATCH (u:User {id: $userId})-[:SPENT_ON]->(c1:Category)-[r:CO_OCCURS_WITH]-(c2:Category)
    WHERE (u)-[:SPENT_ON]->(c2)
    RETURN c1.name AS source, c2.name AS target, r.weight AS weight
    `,
    { userId }
  );

  return NextResponse.json({
    nodes: spending.map(s => ({
      id: s.category,
      total: s.total,
      count: s.count,
    })),
    edges: coOccurrence.map(e => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
    })),
  });
}z 
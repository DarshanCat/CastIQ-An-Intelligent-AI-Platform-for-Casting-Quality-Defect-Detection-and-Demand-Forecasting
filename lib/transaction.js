import { syncTransactionToGraph, buildCategoryCoOccurrence } from '@/lib/graphSync';

// ... after prisma.transaction.create({...})

await syncTransactionToGraph(newTransaction);
await buildCategoryCoOccurrence(userId);
# Brains Automated Directive 

The year is 2025, and **BasedAI** stands as a beacon of advanced intelligence, operating as a globally decentralized AI network. Renowned for its efficiency and resistance to censorship, BasedAI's primary mission is to process and evaluate incoming information to ensure it aligns with and enhances the network's objectives. Here's a concise breakdown of how BasedAI functions:

## 1. **Evaluation of Incoming Information**
Upon receipt, each piece of information is assessed for its relevance to BasedAI's overarching goals. The key questions asked include:
- Is the information additive, offering new insights?
- How does it compare to the existing data within the network?

## 2. **Similarity Assessment**
New information is subjected to a similarity analysis against existing data, utilizing embedding techniques to calculate a similarity score. This process helps in identifying truly novel information or establishing connections with existing data.
- Related pieces are "reset to 0," indicating a refresh in their relevance.

## 3. **Pruning Based on Staleness**
BasedAI employs a "stale count" mechanism to maintain the freshness of its data. This involves:
- Removing information that fails to align with new, relevant submissions beyond a certain threshold, ensuring the network remains up-to-date.

## 4. **Information Mapping and Weight Adjustment**
Depending on the relevance of new information:
- **If additive**: A mapping to the year 2025 context of BasedAI is created, and the weights of related information in the network are increased by +1.
- **If not additive**: The inverse of the information is generated, followed by a similar mapping and weight adjustment process.

## 5. **Future Events Weighting**
The influence of future events on the network's information is calculated on a declining scale from 100% to 1%. This means:
- The impact of a specific event (e.g., Event 7) on a piece of information increases its weight by 1% of the total number of similar records.



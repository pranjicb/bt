# Implementing Graph Mining Algorithms on GPUs based on SISA"
This repository contains the public code for my Bachelor's Thesis

## Problem and Motivation
Graph mining as part of data mining is becoming increasingly important, as it allows one to
find a general solution to many problems from different fields of science, like in chemistry
[14], social sciences [15] and medicine [16]. Graph pattern matching is a subclass of graph
mining, with many applications in different fields [17-20], where the goal is to find subgraphs
corresponding to a certain pattern. Well-known algorithms for these types of problems do not
effectively utilize parallel hardware solutions because of non-straightforward parallelism.
Also, most well-known algorithms are vertex- or edge-centric, which entails huge
complexities for large datasets, such as social media networks (e.g., SNAP dataset [21]).
More and more data gets digitized and this data growth requires more efficient algorithms.


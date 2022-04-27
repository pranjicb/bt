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

## Goal
The goal is to accelerate graph mining algorithms on GPU and to create a CUDA-based
library that will solve different graph related problems efficiently.

## Key Idea
The idea is to use set-centric formulations of graph mining algorithms, like in SISA [1], and
to implement them on CUDA enabled GPUs. Set operations are a crucial part to many graph
mining algorithms and offer rich and simple parallelism at multiple levels and offer
unprecedented improvements [1]. GPUs are a good fit for this problem as they have a lot of
hardware parallelism, provide large memory bandwidth and the programming model allows
programmers to express irregular algorithms (e.g., dynamic parallelism).

## Key Novelty
There have been numerous research efforts towards graph mining problems on GPUs (see
related work section), but most of them follow vertex-centric approaches. Vertex-centric
approaches are not optimal for graph mining problems, as the algorithms require non-local
knowledge of the graph and not only the neighbours of vertices [22].
This bachelor’s thesis will follow the set-centric approach proposed in the SISA paper [1].
Since large parts of graph mining algorithms use many set operations (e.g., in triangle
counting, up to 94% of the runtime is used for set intersections [23]), these operations are a
good target to decrease the overall runtime and complexity of these algorithms.

## Mechanism Overview
1. Surveying the literature and understanding the graph algorithms.
2. Understanding the SISA reference codes [1,32,34].
3. Finding good baseline codes for CPU and GPU implementations for the
graph algorithms to make comparisons.
4. Implementing the SISA algorithms on the GPU and writing a CUDA-based
library.
5. Evaluating the implementations with synthetic and real datasets.
6. Comparing the results with the CPU and GPU baselines.
7. Writing the bachelor’s thesis in the shape of a research paper that could be
published at a top conference.

## Evaluation Methodology
The experiments will be performed on a 3.8 GHz AMD Ryzen 5800X and 16 GB RAM, for
the CPU implementations [24-29], and a NVIDIA GeForce RTX 3070 Ti with 8 GB
GDDR6X global memory for the GPU baselines [2-8,10,11,33] and the GPU+SISA
implementations. We will use the same datasets like in the SISA paper [1] for testing. We will
use the NVIDIA Nsight compute [31] to analyze memory usage, since the data will be
represented in different ways and also look at the efficiency of the kernels. We will look at
the scalability of the algorithms implemented on SISA+GPU, to find optimal grid sizes, and
do the analysis of sparse vs. dense graphs.

## Expected Results
In the SISA paper, algorithmic vs. architectural speedups are discussed [1]. They show that
the algorithmic speedups of SISA depend on the targeted mining algorithm. Those that
already have a high algorithmic speedup will probably keep it that way, while a problem like
triangle counting, where the set-centric non-architectural implementation is slower than the
hand-tuned algorithms, can profit highly of the parallel hardware of a GPU as it was shown in
previous work [23].

## Key Contributions
The main contributions of this Bachelor’s thesis will be as follows:
- **This work will show that GPUs are a good fit for the SISA framework**
-  This work will be the first to implement the set-centric graph mining algorithms
presented in SISA on GPUs.
- This work will create an open-source CUDA-based library with programmer’s
friendly implementations of graph mining algorithms.
- a high-quality paper publishable at a top venue.

## Related Work
**SISA [1]**: This work is heavily based on the SISA paper. In SISA the authors identify and
expose set operations in graph mining algorithms, and then map these algorithms to a
set-centric instruction set architecture extension for graph mining. Also, they use PIM to
accelerate the SISA instructions.
**Pangolin [9]**: Pangolin is a graph pattern mining system on shared memory CPUs and GPUs.
It allows the user to specify eager enumeration search space pruning and customized pattern
classification.
**Peregrine [10]**: Peregrine is a pattern-aware graph mining system that tries to improve graph
pattern algorithms by avoiding the exploration of unnecessary subgraphs.
**FlexMiner [11]**: FlexMiner is a hardware/software graph pattern mining accelerator.
Together with **Pangolin**, the codes are implemented in **GraphMiner [33]**, which we will also
use as GPU baselines.
**FINGERS [12]**: FINGERS is another graph pattern mining accelerator, that is directly
compared to FlexMiner. It exploits fine-grained parallelism at the branch, set and segment
levels during search tree exploration and set operations of pattern-aware graph mining.
**TrieJax [13]**: FINGERS is another graph pattern mining accelerator, that is directly
compared to FlexMiner. It exploits fine-grained parallelism at the branch, set and segment
levels during search tree exploration and set operations of pattern-aware graph mining.
**GPU implementations for specific graph mining problems [2,8].
Well-known CPU implementations for specific graph mining problems [24,29].**

# References 
[1] Maciej Besta, Raghavendra Kanakagiri, Grzegorz Kwasniewski, Rachata
Ausavarungnirum, Jakub Beránek, Konstantinos Kanellopoulos, Kacper Janda, Zur
Vonarburg-Shmaria, Lukas Gianinazzi, Ioanna Stefan, Juan Gómez-Luna, Marcin Copik,
Lukas Kapp-Schwoerer, Salvatore Di Girolamo, Nils Blach, Marek Konieczny, Onur Mutlu,
Torsten Hoefler. 2021. SISA: Set-Centric Instruction Set Architecture for Graph Mining on
Processing-in-Memory Systems. In MICRO 2021.
[2] Yi-Wen Wei, Wei-Mei Chen, Hsin-Hung Tsai. 2021. Accelerating the Bron-Kerbosch
Algorithm for Maximal Clique Enumeration Using GPUs. In IEEE Transactions on Parallel
and Distributed Systems (Volume: 32, Issue: 9, Sept.1 2021).
https://ieeexplore.ieee.org/document/9381690
[3] Mohammad Almasri, Izzat El Hajj, Rakesh Nagi, Jinjun Xiong, Wen-mei Hwu. 2021.
K-Clique Counting on GPUs. https://doi.org/10.48550/arXiv.2104.13209
[4] Yang Hu, Hang Liu, H. Howie Huang. 2018. TriCore: Parallel Triangle Counting on
GPUs. In SC18: International Conference for High Performance Computing, Networking,
Storage and Analysis, November 11-16, 2018.
[5] Li Zeng, Lei Zou, M. Tamer Özsu, Lin Hu, Fan Zhang. 2020. GSI: GPU-friendly
Subgraph Isomorphism. In 2020 IEEE 36th International Conference on Data Engineering
(ICDE), April 20-24, 2020. https://ieeexplore.ieee.org/document/9101348
[6] Afton Geil, Yangzihao Wang, John D. Owens. 2014. WTF, GPU! computing twitter’s
who-to-follow on the GPU. In COSN ‘14: Proceedings of the second ACM conference on
Online social networks. October 2014. Pages 63-68.
https://doi.org/10.1145/2660460.2660481
[7] Thomas Ryan Stovall, Sinan Kockara, Recev Avci. 2014. GPUSCAN: GPU-Based
Parallel Structural Clustering Algorithm for Networks. IEEE Transactions on Parallel and
Distributed Systems (Volume: 26, Issue: 12, Dec. 1 2015).
https://ieeexplore.ieee.org/document/6967853
[8] Yusuke Kozawa, Toshiyuki Amagasa, Hiroyuki Kitagawa. 2017. GPU-Accelerated Graph
Clustering via Parallel Label Propagation. In CIKM ‘17: Proceedings of the 2017 ACM on
Conference on Information and Knowledge Management, November 2017, Pages 567-576.
https://doi.org/10.1145/3132847.3132960
[9] Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali. Pangolin: An Efficient
and Flexible Parallel Graph Mining System on CPU and GPU. PVLDB, 13(8): 1190-1205,
2020. DOI: https://doi.org/10.14778/3389133.3389137
[10] Kasra Jamshidi, Rakesh Mahadasa, Keval Vora. 2020. Peregrine: A Pattern-Aware
Graph Mining System. In EuroSys ‘20: Proceedings of the Fifteenth European Conference on
Computer Systems, April 2020, Article No.: 13, Pages 1-16.
https://doi.org/10.1145/3342195.3387548
[11] Xuhao Chen, Tianhao Huang, Shuotao Xu, Thomas Bourgeat, Chanwoo Chung, Arvind
Arvind. 2021. FlexMiner: A Pattern-Aware Accelerator for Graph Pattern Mining. In 2021
ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA).
https://ieeexplore.ieee.org/document/9499844
[12] Qihang Chen, Boyu Tian, Mingyu Gao. 2022. FINGERS: Exploiting Fine-Grained
Parallelism in Graph Mining Accelerators. In Proceedings of the 27th ACM International
Conference on Architectural Support for Programming Languages and Operating Systems
(ASPLOS ‘22), February 28-March 4, 2022, Lausanne, Switzerland. ACM, New York, NY,
USA, 13 pages. https://doi.org/10.1145/3503222.3507730
[13] Oren Kalinsky, Benny Kimelfeld, Yoav Etsion. 2020. The TrieJax Architecture:
Accelerating Graph Operations Through Relational Joins. In ASPLOS ‘20: Proceedings of
the Twenty-Fifth International Conference on Architectural Support for Programming
Languages and Operating Systems, March 2020, Pages 1217-1231.
https://doi.org/10.1145/3373376.3378524
[14] Ichigaku Takigawa, Hiroshi Mamitsuka. 2012. Graph mining: procedure, application to
drug discovery and recent advances. In Drug Discovery Today (Volume: 18, Issues: 1-2,
January 2013, Pages: 50-57). https://doi.org/10.1016/j.drudis.2012.07.016
[15] Lei Tang, Huan Liu. 2010. Graph Mining Applications to Social Network Analysis. In:
Aggarwal, C., Wang, H. (eds) Managing and Mining Graph Data. Advances in Database
Systems, vol 40. Springer, Boston, MA. https://doi.org/10.1007/978-1-4419-6045-0_16
[16] Cagatay Bilgin, Cigdem Demir, Chandandeep Nagi, Bulent Yener. 2007. Cell-Graph
Mining for Breast Tissue Modeling and Classification. In 2007 29th Annual International
Conference of the IEEE Engineering in Medicine and Biology Society.
https://doi.org/10.1109/IEMBS.2007.4353540
[17] Töpfer A, Marschall T, Bull RA, Luciani F, Schönhuth A, Beerenwinkel N (2014) Viral
Quasispecies Assembly via Maximal Clique Enumeration. PLoS Comput Biol 10(3):
e1003515. https://doi.org/10.1371/journal.pcbi.1003515
[18] Wenfei Fan. 2012. Graph pattern matching revised for social network analysis. In ICDT
‘12: Proceedings of the 15th International Conference on Database Theory, March 2012,
Pages 8-21. https://doi.org/10.1145/2274576.2274578
[19] Aaron Smalter, Jun Huan, Gerald Lushington. 2008. GPM: A graph pattern matching
kernel with diffusion for chemical compound classification. In 2008 8th IEEE International
Conference on BioInformatics and BioEngineering.
https://doi.org/10.1109/BIBE.2008.4696654
[20] Adrian Pearce, Terry Caelli, Walter F. Bischof, Rulegraphs for graph matching in pattern
recognition, Pattern Recognition, Volume 27, Issue 9, 1994, Pages 1231-1247, ISSN
0031-3203, https://doi.org/10.1016/0031-3203(94)90007-8.
[21] Stanford Large Network Dataset Collection. https://snap.stanford.edu/data/
[22] Diane J Cook and Lawrence B Holder. 2006. Mining graph data. John Wiley & Sons.
[23] Shuo Han, Lei Zou, and Jeffrey Xu Yu. 2018. Speeding Up Set Intersections in Graph
Algorithms using SIMD Instructions. In Proceedings of the 2018 International Conference on
Management of Data. ACM, 1587–1602.
[24] CPU Implementation of Bron-Kerbosch by Google.
https://google.github.io/or-tools/cpp_graph/cliques_8h_source.html
[25] Coen Bron and Joep Kerbosch. 1973. Algorithm 457: finding all cliques of an undirected
graph. Commun. ACM 16, 9 (1973), 575–577.
[26] Maximilien Danisch, Oana Balalau, and Mauro Sozio. 2018. Listing k-cliques in sparse
real-world graphs. In Proceedings of the 2018 World Wide Web Conference on World Wide
Web. International World Wide Web Conferences Steering Committee, 589–598
[27] Luigi P Cordella, Pasquale Foggia, Carlo Sansone, and Mario Vento. 2004. A (sub)
graph isomorphism algorithm for matching large graphs. IEEE transactions on pattern
analysis and machine intelligence 26, 10 (2004), 1367–1372.
[28] Maciej Besta, Raghavendra Kanakagiri, Harun Mustafa, Mikhail Karasikov, Gunnar
Rätsch, Torsten Hoefler, and Edgar Solomonik. 2020. Communication-efficient jaccard
similarity for high-performance distributed genome comparisons. In 2020 IEEE International
Parallel and Distributed Processing Symposium (IPDPS). IEEE, 1122–1132.
[29] Raymond Austin Jarvis and Edward A Patrick. 1973. Clustering using a similarity
measure based on shared near neighbors. IEEE Transactions on computers 100, 11 (1973),
1025–1034
[30] GPU implementation for WTF, GPU!
https://gunrock.github.io/docs/#/gunrock/gunrock_applications
[31] NVIDIA Nsight Compute https://developer.nvidia.com/nsight-compute
[32] GraphMineSuite https://graphminesuite.spcl.inf.ethz.ch/
[33] GraphMiner https://github.com/chenxuhao/GraphMiner
[34] Maciej Besta, Zur Vonarburg-Shmaria, Yannick Schaffner, Leonardo Schwarz,
Grzegorz Kwasniewski, Lukas Gianinazzi, Jakub Beranek, Kacper Janda, Tobias
Holenstein, Sebastian Leisinger, et al. 2021. GraphMineSuite: Enabling High-
Performance and Programmable Graph Mining Algorithms with Set Algebra.
VLDB (2021).

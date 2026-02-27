# Network Traffic Classification Architecture

# ET-BERT
ET-BERT is a method for learning datagram contextual relationships from encrypted traffic, which could be directly applied to different encrypted traffic scenarios and accurately identify classes of traffic. First, ET-BERT employs multi-layer attention in large scale unlabelled traffic to learn both inter-datagram contextual and inter-traffic transport relationships. Second, ET-BERT could be applied to a specific scenario to identify traffic types by fine-tuning the labeled encrypted traffic on a small scale.

## BERT
<img src="./image/image04.png" width="80%" />

# Trident
Trident addresses two primary challenges in network traffic classification: (i) the detection of fine-grained, emerging attacks, and (ii) incremental updates and adaptations. To address these issues, the researchers reformulate the identification of known and new classes as multiple, independent one-class learning tasks, thereby decoupling model capabilities. Based on this approach, Trident is designed as a universal framework for fine-grained detection of unknown encrypted traffic. The framework includes three key modules: tSieve (for traffic profiling), tScissors (for determining outlier thresholds), and tMagnifier (for clustering), each with support for custom configurations. Experiments conducted on four widely-used network trace datasets indicate that Trident achieves significant performance improvements over 16 state-of-the-art (SOTA) methods. Further evaluations, such as concept drift handling and assessments of computational overhead and parameter efficiency, demonstrate the stability, scalability, and practicality of the Trident framework.

## AutoEncoder
<img src="./image/image05.png" width="80%" />

## RNN
<img src="./image/image06.png" width="80%" />

## GNN
<img src="./image/image07.png" width="80%" />

# MH-Net
MH-Net addresses the shortcomings of traditional byte-level analysis by introducing a novel classification approach that utilizes multi-view heterogeneous traffic graphs to represent detailed relationships among traffic bytes. MH-Net creates multiple types of traffic units by aggregating different numbers of bits, resulting in multi-view traffic graphs with varying levels of information granularity. By modeling various byte correlations—such as those between header and payload—MH-Net introduces heterogeneity to the traffic graph, which substantially boosts model performance. Additionally, it employs multi-task contrastive learning to reinforce the robustness of traffic unit representations. Experiments on the ISCX and CIC-IoT datasets, considering both packet-level and flow-level classification, demonstrate that MH-Net consistently outperforms numerous state-of-the-art methods.

## GNN + RNN
<img src="./image/image08.png" width="80%" />

# LiteNet
LiteNet offers a comprehensive, end-to-end workflow for training, optimizing, and deploying neural network models for Network Traffic Classification (NTC). Its pipeline incorporates SHAP-based feature selection, semi-structured sparse pruning, quantization to FP16 or INT8, and conversion into a TensorRT engine to enable high-performance inference.

## CNN
<img src="./image/image09.png" width="80%" />

## Compression Techniques
<img src="./image/image10.png" width="80%" />
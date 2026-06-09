
# PCAP Method Part

 3.1 Problem Formulation
 
  We study weakly supervised fine-grained video anomaly detection, where only video-level category labels are available during training. Given a training set $\mathcal{D}={(V_i, y_i)}_{i=1}^{N}$, each video ($V_i$) is associated with a video-level label $y_i \in \{0,1,\dots,C\},$ where (0) denotes the normal class and ($1,\dots,C$) denote different anomaly categories. No frame-level or segment-level temporal annotations are provided.

  For each video (V), we uniformly sample (T) representative frames or snippets: $V={x_t}_{t=1}^{T}.$ The goal is to predict the fine-grained video-level anomaly category while also producing segment-level anomaly scores for localization evaluation. Unlike conventional binary weakly supervised video anomaly detection, our setting requires the model to distinguish not only whether a video is abnormal, but also which anomaly category it belongs to.

  This task is challenging because video-level supervision does not indicate which temporal segments contain abnormal events. Moreover, different anomaly categories may share visually similar backgrounds, human actions, or motion patterns. Therefore, directly learning visual class prototypes from weak labels can easily suffer from noisy background segments and inter-class confusion.

  To address this problem, we propose a prompt-aligned class prototype framework that uses frozen CLIP textual priors to guide class prototype learning under weak supervision.
 
 3.2 Framework Overview
 
  The proposed framework consists of four main components:
  1. Appearance-motion feature encoding, which extracts frame-level appearance features using a frozen CLIP image encoder and derives motion features through temporal feature differences.
  2. Temporal feature enhancement, which fuses appearance and motion cues and enhances temporal representations through a lightweight temporal head.
  3. Learnable semantic prompt bank, which constructs one text prompt for each class using shared learnable context tokens and a frozen CLIP text encoder.
  4. Prompt-aligned class prototype module, which learns one visual prototype for each class and aligns these prototypes with their corresponding textual prompt embeddings.

  Given a sampled video sequence, the model first extracts appearance features from representative frames. Motion features are computed by feature-level temporal differences. The appearance and motion features are then fused and enhanced to obtain segment-level visual representations. For each class, a learnable text prompt is encoded by the frozen CLIP text encoder to produce a semantic class embedding. Each class also owns one learnable visual prototype.

  Segment-level class logits are computed by comparing visual features with both semantic prompt embeddings and visual prototypes. Finally, top-($k$) multiple instance learning aggregates segment-level predictions into video-level class scores.

 3.3 Appearance-Motion Feature Encoding

  For each sampled frame or snippet (x_t), we use the frozen CLIP image encoder (E_I(\cdot)) to extract an appearance representation: 
 $$\mathbf{a}_t = \text{Norm}(E_I(x_t)),$$
  where ($\text{Norm}(\cdot)$) denotes ($l_2$) normalization.

  Instead of introducing an additional optical-flow network or encoding frame differences directly, we compute motion cues in the CLIP feature space. Specifically, the motion representation is obtained by the temporal difference between adjacent appearance features:
  
$
  \mathbf{m}_t = \text{Norm}(\mathbf{a}t - \mathbf{a}{t-1}),
$

  where ($\mathbf{m}_1$) is initialized as a zero vector or copied from the first valid difference.

  This feature-level difference provides a simple and efficient motion descriptor. It avoids additional motion encoders while still capturing temporal changes between adjacent sampled frames. Since the CLIP image encoder is frozen, this design also keeps the trainable part of the model lightweight.

  The appearance and motion features are concatenated and projected into a unified feature space:
$
  \mathbf{u}_t = \phi_f([\mathbf{a}_t ; \mathbf{m}_t]),
$

  where ($[\cdot ; \cdot]$) denotes feature concatenation and ($\phi_f(\cdot)$) is a trainable projection layer.

  To incorporate temporal context, the fused sequence (${\mathbf{u}_{t=1}^{T}}$) is further processed by a temporal enhancement head:
$
  \mathbf{z}{1:T} = H{\theta}(\mathbf{u}{1:T}),
$

  where (H{\theta}) can be implemented as a lightweight temporal convolutional head, temporal Transformer head, or other sequence modeling module. The output ($\mathbf{z}_t$) denotes the enhanced visual representation of the (t)-th segment.

  3.4 Learnable Semantic Prompt Bank

  To introduce semantic class priors, we construct a text prompt for each class. Instead of manually designing multiple templates for every anomaly category, we use a single shared prompt structure with learnable context tokens:
$$
  P_c = [\mathbf{v}_1][\mathbf{v}_2]\dots[\mathbf{v}_M]\ \text{a surveillance video of}\ [\text{CLASS}_c],
$$

  where ($[\mathbf{v}_1],\dots,[\mathbf{v}_M]$) are learnable context tokens shared by all classes, and ($[\text{CLASS}_c]$) is the textual name of class (c).

  The prompt ($P_c$) is fed into the frozen CLIP text encoder ($E_T(\cdot)$):
$$
  \mathbf{e}_c = \text{Norm}(E_T(P_c)).
$$

  Here, ($E_T(\cdot)$) is frozen, while the context tokens are trainable. This design allows the model to adapt textual prompts to the weakly supervised video anomaly detection task without manually engineering class-specific templates.

  The resulting text embedding ($\mathbf{e}_c$) serves as a semantic anchor for class (c). Compared with randomly initialized class prototypes, these prompt embeddings provide category-level semantic structure from CLIP’s pretrained vision-language space.

  3.5 Prompt-Aligned Class Prototype Module

  For each class (c), including the normal class, we learn one visual class prototype:

$$
  \mathbf{p}_c \in \mathbb{R}^{d}.
$$

  The prototype ($\mathbf{p}_c$) represents the visual anchor of class (c) in the enhanced feature space. To avoid purely random prototype learning, each prototype is initialized from or aligned with its corresponding text prompt embedding ($\mathbf{e}_c$).

  Given a segment-level visual representation ($\mathbf{z}t$), we compute its similarity to each class prototype:
$$
  s^{p}{t,c} = \tau_p \cdot \cos(\mathbf{z}_t, \mathbf{p}_c),
$$

  where ($\tau_p$) is a temperature parameter.

  To make prompt learning participate directly in prediction rather than only serving as a regularizer, we also compute the similarity between the visual feature and the semantic prompt embedding:
$$
  s^{e}_{t,c} = \tau_e \cdot \cos(\mathbf{z}_t, \mathbf{e}_c).
$$

  The final segment-level class logit is obtained by combining prototype-based and prompt-based similarities:
$$
  s_{t,c} = s^{p}{t,c} + \alpha s^{e}{t,c},
$$
  where ($\alpha$) controls the contribution of the semantic prompt branch.

  This design gives the model two complementary sources of class information. The visual prototype ($\mathbf{p}_c$) adapts to the weakly supervised training data, while the prompt embedding ($\mathbf{e}_c$) provides a stable semantic prior from CLIP. Their alignment encourages each learned prototype to remain semantically meaningful, which is especially important when training only with video-level labels.

  The segment-level class probability is then computed by:
$$
  P_{t,c} = \frac{\exp(s_{t,c})}{\sum_{j=0}^{C}\exp(s_{t,j})}.
$$

  3.6 Top-(k) Multiple Instance Learning

  Since only video-level labels are available, we use top-(k) multiple instance learning to aggregate segment-level predictions into video-level scores. For each class (c), we select the top-(k) segment logits:
$$
  \mathcal{T}c = \text{TopK}({s{t,c}}{t=1}^{T}, k).
$$

  The video-level score for class (c) is computed as: 
$$
  S_c = \frac{1}{k}\sum_{t \in \mathcal{T}c} s{t,c}.
$$

  The video-level class distribution is:
$$

  P_c^{video} = \frac{\exp(S_c)}{\sum_{j=0}^{C}\exp(S_j)}.
$$

  Given the video-level ground-truth label (y), the fine-grained classification loss is:
$$
  \mathcal{L}_{cls} = -\log P_y^{video}.
$$

  This top-(k) strategy is suitable for weakly supervised anomaly detection because abnormal events usually occur in only a subset of video segments. Instead of averaging predictions over the entire video, top-(k) pooling allows the model to focus on the most discriminative temporal segments for each class.

  3.7 Prompt-Prototype Alignment Loss

  To explicitly connect semantic prompts and visual prototypes, we introduce a prompt-prototype alignment loss:
$$
  \mathcal{L}_{align} = \frac{1}{C+1}\sum{c=0}^{C}\left(1-\cos(\mathbf{p}_c,\mathbf{e}_c)\right).
$$

  This loss encourages each visual prototype to stay close to its corresponding semantic prompt embedding. As a result, the learned prototypes are not merely unconstrained visual class centers, but semantic class anchors guided by CLIP textual priors.

  This alignment is particularly useful under weak supervision. Since the model does not know which segments are truly abnormal, prototypes learned only from video-level labels may be affected by normal background frames. The semantic prompt embeddings provide additional class-level constraints and help reduce prototype drift.

  3.8 Prototype Separation Loss

  Fine-grained anomaly categories often share similar visual patterns. For example, fighting, abuse, and riot may all involve human interactions and crowded scenes. To reduce inter-class confusion, we encourage different class prototypes to be separated from each other.

  The prototype separation loss is defined as:

$$

  \mathcal{L}_{sep}=

  

  \frac{1}{C(C+1)}

  \sum_{i=0}^{C}

  \sum_{\substack{j=0 \ j\neq i}}^{C}

  \max(0, \cos(\mathbf{p}_i,\mathbf{p}_j)-\delta),

 $$
  where ($\delta$) is a margin hyperparameter.

  

  This loss penalizes excessive similarity between different class prototypes. It improves the discriminability of prototype representations and strengthens fine-grained category boundaries.
  
  3.9 Overall Objective

  The overall training objective is:
$$
  \mathcal{L} =

  

  \mathcal{L}_{cls}

  +

  \lambda_{align}\mathcal{L}_{align}

  +

  \lambda_{sep}\mathcal{L}_{sep},

$$

  where ($\lambda_{align}$) and ($\lambda_{sep}$) control the contributions of the alignment and separation losses.

  Only the projection layer, temporal enhancement head, learnable context tokens, and class prototypes are optimized during training. Both CLIP image encoder and CLIP text encoder remain frozen. This training strategy preserves the pretrained visual-language knowledge of CLIP while adapting a small number of task-specific parameters to weakly supervised fine-grained anomaly classification.

  3.10 Inference

  During inference, a test video is uniformly sampled and processed by the same feature encoding and temporal enhancement pipeline. The model computes segment-level class probabilities ($P_{t,c}$) and video-level class scores ($S_c$).

  The predicted video-level category is:
$$

  \hat{y} = \arg\max_{c \in {0,\dots,C}} P_c^{video}.
$$

  For fine-grained classification evaluation, we use the video-level class probabilities over the normal and anomaly categories.

  For anomaly localization evaluation, the segment-level abnormal score can be derived from the class probabilities without introducing an additional detection head:
$$
  A_t = 1 - P_{t,0},
$$
  or equivalently,
$$

  A_t = \max_{c=1}^{C} P_{t,c}.
$$
  This allows the same model to perform fine-grained anomaly classification and produce segment-level anomaly scores for AUC and AP evaluation.

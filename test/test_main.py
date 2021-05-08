import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

import unittest
from phraseg import *


class Test(unittest.TestCase):

    def testFile(self):
        phraseg = Phraseg("./smailltext")
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testIDF(self):
        phraseg = Phraseg("./smailltext")
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)
        phraseg = Phraseg("./smailltext")
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testEng(self):
        phraseg = Phraseg('''
        The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.
        Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
        End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].
        To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testEngLong(self):
        phraseg = Phraseg('''
         Attention Is All You Need
         Ashish Vaswani∗ Google Brain
        avaswani@google.com
        Llion Jones∗ Google Research
        llion@google.com
        Noam Shazeer∗ Google Brain
        noam@google.com
        Niki Parmar∗ Google Research
        nikip@google.com
        Jakob Uszkoreit∗
        Google Research
        usz@google.com
        Aidan N. Gomez∗ † University of Toronto aidan@cs.toronto.edu
        Łukasz Kaiser∗ Google Brain
        lukaszkaiser@google.com
        Illia Polosukhin∗ ‡ illia.polosukhin@gmail.com
        Abstract
        The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
        1 Introduction
        Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and
        ∗Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.
        †Work performed while at Google Brain. ‡Work performed while at Google Research.
         31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
        arXiv:1706.03762v5 [cs.CL] 6 Dec 2017
        
        transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
        Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.
        Attention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.
        In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.
        2 Background
        The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.
        Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
        End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].
        To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].
        3 Model Architecture
        Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1,...,xn) to a sequence of continuous representations z = (z1,...,zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.
        The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.
        2
        
         Figure 1: The Transformer - model architecture.
        3.1 Encoder and Decoder Stacks
        Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.
        Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.
        3.2 Attention
        An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
        3
        
        Scaled Dot-Product Attention Multi-Head Attention
          Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.
        3.2.1 Scaled Dot-Product Attention
        We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension d , and values of dimension d . We compute the dot products of the
        k√v
        query with all keys, divide each by dk, and apply a softmax function to obtain the weights on the
        values.
        In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:
        QKT
        Attention(Q, K, V ) = softmax( √
        The two most commonly used attention functions are additive attention [2], and dot-product (multi-
        dk
        plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
        of √
        . Additive attention computes the compatibility function using a feed-forward network with
        1
        dk
        a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
        much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
        While for small values of dk the two mechanisms perform similarly, additive attention outperforms
        dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
        .
        dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has 41
        extremely small gradients . To counteract this effect, we scale the dot products by √ 3.2.2 Multi-Head Attention
        dk
        Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv -dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.
        4To illustrate why the dot products get large, assume that the components of q and k are independent random
        variables with mean 0 and variance 1. Then their dot product, q · k = 􏰀dk qiki, has mean 0 and variance dk. i=1
        )V (1)
         4
        
        Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.
        MultiHead(Q, K, V ) = Concat(head1, ..., headh)W O where headi = Attention(QWiQ , K WiK , V WiV )
        Where the projections are parameter matrices WiQ ∈ Rdmodel ×dk , WiK ∈ Rdmodel ×dk , WiV ∈ Rdmodel ×dv O hdv ×dmodel
        .
        In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.
        3.2.3 Applications of Attention in our Model
        The Transformer uses multi-head attention in three different ways:
        • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].
        • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
        • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2.
        3.3 Position-wise Feed-Forward Networks
        In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
        FFN(x) = max(0, xW1 + b1 )W2 + b2 (2)
        While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff =2048.
        3.4 Embeddings and Softmax
        Similarly to other sequence transduction models, we use learned embeddings to convert the input
        tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transfor-
        mation and softmax function to convert the decoder output to predicted next-token probabilities. In
        our model, we share the same weight matrix between the two embedding layers and the pre-softmax
        andW ∈R
        linear transformation, similar to [30]. In the embedding layers, we multiply those weights by
        3.5 Positional Encoding
        √
        dmodel.
        Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the
        5
        
        Table1: Maximumpathlengths,per-layercomplexityandminimumnumberofsequentialoperations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.
         Layer Type
        Self-Attention
        Recurrent
        Convolutional Self-Attention (restricted)
        Complexity per Layer
        O(n2 · d)
        O(n · d2) O(k · n · d2) O(r · n · d)
        Sequential Maximum Path Length Operations
        O(1) O(1) O(n) O(n) O(1) O(logk(n)) O(1) O(n/r)
          tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].
        In this work, we use sine and cosine functions of different frequencies: PE(pos,2i) =sin(pos/100002i/dmodel)
        PE(pos,2i+1) =cos(pos/100002i/dmodel)
        where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of PEpos.
        We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.
        4 Why Self-Attention
        In this section we compare various aspects of self-attention layers to the recurrent and convolu- tional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈ Rd, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we
        consider three desiderata.
        One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
        The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.
        As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in
        6
        
        the input sequence centered around the respective output position. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work.
        A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably, to O(k · n · d + n · d2). Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.
        As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.
        5 Training
        This section describes the training regime for our models.
        5.1 Training Data and Batching
        We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source- target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.
        5.2 Hardware and Schedule
        We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).
        5.3 Optimizer
        We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10 rate over the course of training, according to the formula:
        −9
        . We varied the learning
        lrate=d−0.5 ·min(step_num−0.5,step_num·warmup_steps−1.5) (3) model
        This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.
        5.4 Regularization
        We employ three types of regularization during training:
        Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1.
        7
        
        Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
         Model
        ByteNet [18]
        Deep-Att + PosUnk [39]
        GNMT + RL [38]
        ConvS2S [9]
        MoE [32]
        Deep-Att + PosUnk Ensemble [39] GNMT + RL Ensemble [38] ConvS2S Ensemble [9] Transformer (base model) Transformer (big)
        BLEU
        EN-DE EN-FR 23.75
        Training Cost (FLOPs)
           24.6 25.16 26.03
        26.30 26.36 27.3 28.4
        39.2 39.92 40.46 40.56 40.4 41.16 41.29 38.1 41.8
        EN-DE
        2.3 · 1019 9.6 · 1018 2.0 · 1019
        1.8 · 1020 7.7 · 1019
        EN-FR
        1.0 · 1020 1.4 · 1020 1.5 · 1020 1.2 · 1020 8.0 · 1020 1.1 · 1021 1.2 · 1021
          Label Smoothing During training, we employed label smoothing of value εls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.
        6 Results
        6.1 Machine Translation
        On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.
        On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.
        For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].
        Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained
        5 single-precisionfloating-pointcapacityofeachGPU .
        6.2 Model Variations
        To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.
        In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
        5We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively. 8
        3.3 · 1018 2.3 · 1019
          
        Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.
         N
        base 6 (A)
        (B)
        d d h d d P ε train PPL BLEU params
        model ff k v drop ls steps 512 2048 8 64 64 0.1 0.1 100K
        (dev) (dev) ×106 4.92 25.8 65 5.29 24.9
        5.00 25.5
        4.91 25.8
        5.01 25.4
        5.16 25.1 58 5.01 25.4 60 6.11 23.7 36 5.19 25.3 50 4.88 25.5 80 5.75 24.5 28 4.66 26.0 168 5.12 25.4 53 4.75 26.2 90 5.77 24.6
        4.95 25.5
        4.67 25.3
        5.47 25.7
        4.92 25.7
        4.33 26.4 213
          (C) 256
        1024 128
        1024 4096
        1 512 4 128
        512 128 32 16
        32 128
        16 32
        32 16 16 32
          2 4 8
        32
         (D)
        (E)
        big 6
        0.0 0.2
        0.0
        0.2 positional embedding instead of sinusoids
        1024 4096 16 0.3
        300K
           Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)
         Parser
        Vinyals & Kaiser el al. (2014) [37] Petrov et al. (2006) [29]
        Zhu et al. (2013) [40]
        Dyer et al. (2016) [8] Transformer (4 layers)
        Zhu et al. (2013) [40] Huang & Harper (2009) [14] McClosky et al. (2006) [26] Vinyals & Kaiser el al. (2014) [37] Transformer (4 layers) Luong et al. (2015) [23] Dyer et al. (2016) [8]
        Training
        WSJ only, discriminative WSJ only, discriminative WSJ only, discriminative WSJ only, discriminative WSJ only, discriminative semi-supervised semi-supervised semi-supervised semi-supervised semi-supervised multi-task
        generative
        WSJ 23 F1
        88.3 90.4 90.4 91.7 91.3 91.3 91.3 92.1 92.1 92.7 93.0 93.3
              In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.
        6.3 English Constituency Parsing
        To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural
        9
        
        constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].
        We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.
        We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3 for both WSJ only and the semi-supervised setting.
        Our results in Table 4 show that despite the lack of task-specific tuning our model performs sur- prisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].
        In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the Berkeley- Parser [29] even when training only on the WSJ training set of 40K sentences.
        7 Conclusion
        In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
        For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.
        We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.
        The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.
        Acknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.
        References
        [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
        [2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.
        [3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.
        [4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.
        [5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.
        [6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.
        10
        
        [7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.
        [8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.
        [9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolu- tional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.
        [10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.
        [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im- age recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.
        [12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.
        [13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
        [14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832–841. ACL, August 2009.
        [15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.
        [16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.
        [17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.
        [18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko- ray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.
        [19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.
        [20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
        [21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint
        arXiv:1703.10722, 2017.
        [22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.
        [23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.
        [24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention- based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.
        [25] MitchellPMarcus,MaryAnnMarcinkiewicz,andBeatriceSantorini.Buildingalargeannotated corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.
        [26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152–159. ACL, June 2006.
        11
        
        [27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.
        [28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.
        [29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July 2006.
        [30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.
        [31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.
        [32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.
        [33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi- nov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.
        [34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates, Inc., 2015.
        [35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.
        [36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.
        [37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.
        [38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.
        [39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.
        [40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), pages 434–443. ACL, August 2013.
        12
        
        Input-Input Layer5
        Attention Visualizations
                           Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.
        13
        majority of American governments have passed new laws since 2009 making the registration or voting process more difficult
        majority
        of
        American governments have
        It It
        is is
        in in this this spirit spirit that that aa
        <EOS> <pad> <pad> <pad> <pad> <pad> <pad>
        <EOS> <pad> <pad> <pad> <pad> <pad> <pad>
        ..
        passed new
        laws
        since
        2009 making
        the registration or
        voting process more difficult
        
                                                                                                                                                                                                                                                                                                                                                       Input-Input Layer5
                    Input-Input Layer5
        Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.
        14
        The The The The Law Law Law Law will will will will never never never never be be be be perfect perfect perfect perfect
        ,,
        ,,
        but but
        its its application application should should
        be be
        just just --
        this this
        is is
        what what
        we we
        are are missing missing ,,
        in in
        my my opinion opinion ..
        <EOS> <EOS> <pad> <pad>
        but but
        its its application application should should
        be be
        just just --
        this this
        is is
        what what
        we we
        are are missing missing ,,
        in in
        my my opinion opinion ..
        <EOS> <EOS> <pad> <pad>
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Input-Input Layer5
                   Input-Input Layer5
        Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.
        15
        The The The The
        Law Law Law Law
        will will will will
        never never never never
        be be be be
        perfect perfect perfect perfect ,,,,
        but but but but
        its its its its application application application application should should should should
        be be be be
        just just just just ----
        this this this this
        is is is is
        what what what what
        we we we we
        are are are are missing missing missing missing ,,,,
        in in in in
        my my my my opinion opinion opinion opinion ....
        <EOS> <EOS> <EOS> <EOS> <pad> <pad> <pad> <pad>

        ''')
        result = phraseg.extract(merge_overlap=True,
                                      sent="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.")
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testChi(self):
        phraseg = Phraseg('''
        Wiki（聆聽i/ˈwɪkiː/）是在全球資訊網上開放，且可供多人協同創作的超文字系統，由沃德·坎寧安於1995年首先開發。沃德·坎寧安將wiki定義為「一種允許一群用戶用簡單的描述來建立和連接一組網頁的社會計算系統」[1]。
        有些人認為[2]，Wiki系統屬於一種人類知識的網路系統，讓人們可以在web的基礎上對Wiki文字進行瀏覽、建立和更改，而且這種建立、更改及發布的成本遠比HTML文字小。與此同時，Wiki系統還支援那些面向社群的協作式寫作，為協作式寫作提供了必要的幫助。最後Wiki的寫作者自然構成了一個社群，Wiki系統為這個社群提供了簡單的交流工具。與其它超文字系統相比，Wiki有使用簡便且開放的特點，有助於在一個社群內共享某個領域的知識。
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testChiLong(self):
        phraseg = Phraseg('''
        好看 真的好看

一種十年磨一劍的感覺

基本上是三本柱 美國隊長 鋼鐵人 雷神索爾的電影 XD

一開始鷹眼開開心心的教女兒射箭 就知道沒五分鐘就要剩粉了QQ

接著是鋼鐵人的部分 雖然跟預告一樣 也知道他不會就掛在太空船

只是還是擔心 直到驚奇隊長閃閃發光的把鋼鐵人閃醒XD

飛回地球 鋼鐵人與美隊的小爭執 我們真的一起輸了 有種揪心感

就算後來飛去找薩諾斯 再次上演手套爭奪戰

索爾這次飛快的砍掉手(復3沒人有刀來著XD

手套掉下來 只是沒了寶石 沒了寶石 沒了寶石 

當下跟著影中人一起錯愕了一下

薩老緩緩說出我用寶石滅了寶石

涅布拉還默默補充我爸不會說謊...

雷神很雷的就把薩諾斯的頭給剁了 想問甚麼也沒得問了= =(雖然是有呼應復3拉

接著就進入哀傷的五年後

美隊在開導大家

黑寡婦在當代理局長

驚奇隊長又飛去宇宙了 所以只回來打了一架

發現沒辦法逆轉現況之後就又飛走了 說好的逆轉無限呢-.-

然後美隊跟黑寡婦聊天 接著被老鼠救出來的蟻人跑來

開啟了這集最重要的量子空間回過去

然後去找了幸福美滿的東尼QQ

可以理解東尼不想幫忙 因為他的模型還沒做出來(誤

只是看著小蜘蛛的照片 讓他又爬起來努力的跑量子模型

另一邊美隊去找了可以維持浩克先生的班納

然後各種搞笑XD

蟻人各種變 有點擔心他回不來 哈哈

這時 東尼帶著金頭腦跟盾牌回來啦~~~

隊長還是要拿著圓盾比較有FU

火箭跟浩克先生去找了雷神 喔 是肥神索爾回來

索爾真的很懊悔 從媽媽救不到 爸爸在眼前消失 弟弟被薩諾斯掛了

朋友因為自己沒有給薩諾斯致命一擊而灰飛煙滅了一半

五年來不斷酗酒走不出來也是蠻合理的

只是這集三本柱最雷的就是他了XD

然後黑寡婦去找到了變成浪人的鷹眼 把他帶回來

回到總部 分隊完

穿好量子戰衣

出發

美國隊長 + 鋼鐵人 + 蟻人 + 浩克先生-> 復1的紐約

火箭浣熊 + 肥神 -> 雷神2的阿斯嘉

涅布拉 + 戰爭機器 + 鷹眼 + 黑寡婦 -> 星際異工隊1

總覺得最後一個分組怪怪的 怎麼會給三個人類去外星球搶東西= =





復1的紐約

浩克看到自己以前的樣子 學著砸了一些東西的場景真的很逗趣XD

然後他去找當時守護寶石的古一 開始了一串辯論

浩克先生這集真的沒甚麼武戲 都在講話

然後隊長走進電梯 場景跟美2一模一樣

本來以為要開扁了 結果居然正直的美隊以一句Hail Hydra輕鬆A走了權杖

順便致敬漫畫 XD

也順勢解釋為什麼復1之後權杖又跑去九頭蛇那 做實驗做出了緋紅女巫&快銀

鋼鐵人&蟻人 本來要搶到空間魔方 結果來一個浩克討厭走樓梯XD

寶石就這樣被洛基A走順便烙跑

然後隊長走著走著遇到了自己

以為未來隊長是洛基的過去隊長 開始對A

然後又來一句隊長經典台詞

I can do this all day , 我知道XD

終究是自己了解自己 以一句巴奇還活著影響自己 順利尻飛過去隊長

然後權杖戳下去 帶走權杖

會合後再度前往1970 去找還在神盾局的魔方&皮姆

這邊偷東西就沒甚麼意外 再有意外就演不完了XD

這邊的主軸 給了鋼鐵人與爸爸的相會

為人父的鋼鐵人有太多事想跟爸爸講 終於圓夢

以及隊長看卡特 為結局埋了伏筆

原來東尼本來不叫東尼XD

阿斯嘉這邊

肥神各種雷 回到過去還在找酒

然後看到媽媽就忍不住想要回去找媽媽

完全不顧大局 單就這個行為真的很雷

只是這集就是給三本柱圓夢的 當然要給肥神回去找媽媽聊天

搶寶石就交給火箭 他也很輕鬆地搶到了寶石

然後圓完雷神的夢

雷神把喵喵槌叫了過來 帶回未來(當下OS是他把槌子拿走了 待會過去的雷神要怎麼用啊

不過後面就知道雷神回到過去最重要的其實就是把槌子拿走 哈哈哈哈





星際異攻隊這邊

鷹眼跟黑寡婦搭著太空船飛到了佛米爾星

話說 哪來的太空船

而且給兩個人類四處太空旅行 每個星球都有氧氣來著= =??

然後又遇到了守門員紅骷顱

接著開始誰要跳 鷹眼你射箭阻止黑寡婦的時候壓根就想炸死他吧XDDD

然後兩個人一起跳下去的時候

我以為會出現老梗之 互相珍惜所愛的人 並且能為對方而死

才是得到靈魂寶石的正確方法

結果是出現另一個老梗 下面的人鬆手往下跳

黑寡婦真的死了!? 靈魂寶石GET (當下想說不是有個人電影 後來才知道是前傳

涅布拉跟戰爭機器

一拳KO星爵就拿到寶石了

只是本篇最雷的涅布拉不小心跟過去涅布拉連結在一起

然後就被抓了 各種劇透給2014薩諾斯

知道復仇者在過去偷寶石之後 決定跟著到未來

因為寶石已經有人幫忙收集好了

只是 皮姆粒子不是只有一個人來回

後來是怎麼讓2014涅布拉去2023 又可以開啟量子通道給薩諾斯大軍過來呢??

畫面回到2023 過去不管多久 未來其實只過了一秒

所以也能解釋涅布拉被換了也不知道 因為未來沒有比較晚回來



接著把寶石裝上手套 (說好的手套要由矮人做才能發揮威力勒~~~

關在小房間 準備彈指

為啥不把捏布拉也關進來 放他在外面開量子通道 嘖嘖

順利彈指 浩克先生差點就掛了

外面鳥開始飛 鷹眼的手機響了

蟻人開始開心的說我們做到了!

只是看電影的大家都知道 糟了

接著總部就被各種轟炸

薩諾斯坐在外面等著倖存的復仇者出來

然後 三本柱 帥氣登場

開始三英戰呂布

只是薩諾斯就算沒手套也是很強

肥神應該是因為變太肥了 感覺很弱

還被壓著差點被自己的風暴毀滅者戳死

這時

隊長

舉起喵喵槌拉!!!!!!!

超帥 不解釋 看得我都泛淚了

然後帥打一波 還有槌子+盾牌合體技

只是帥打一波之後又被屌虐一波回來

連盾牌都被打飛一半

還被嘴+大軍壓境

隊長渾身傷

只默默拉緊盾牌的帶子

再次地站了起來

雖千萬人吾往矣阿!!!!

超帥的 帥到我又泛淚了

這時絕望感來到最高

一句 Cap do you copy! On your left

然後火圈開始冒出

終於

全部人都回來了

Avengers assemble

接著開始各種秀

鋼鐵小辣椒 女武神天馬 黃蜂女 黑豹 奇異博士 小蜘蛛

手套接力賽其實只是秀大家技能的時候

隊長跟索爾交換武器 索爾還說你拿小的就好

葛魔拉怒踢星爵蛋蛋 涅布拉說你只有他跟樹可以選

然後緋紅女巫屌打2014薩諾斯 只是2014薩諾斯表示你誰啊XD

2014薩諾斯表示: 我啥事都沒做 干我屁事XD

被女巫打得受不了了 惱羞開戰艦砲轟地面

還在想這台煩人的戰艦怎麼辦 就有道紅光來解決一切

驚奇隊長終於飛回來了- -

都快打完了還不回來

打飛戰艦後

給驚奇隊長跟薩諾斯單挑一波

寶石原來還可以單拿起來轟人 只能說薩諾斯真的很會用寶石

接著就是東尼看了正在治水的奇異博士

博士只默默伸出食指

然後東尼就衝去搶手套了

本來以為手套又搶失敗了

薩諾斯拉好手套 這次沒人阻止他了

薩諾斯: I'm Inevitable

!? 啥事都沒發生

只見東尼舉起手 寶石已經全都A過來了











                           東尼 : I am Iron Man











一彈指 一切就結束了

只是鋼鐵人 就這樣華麗的謝幕了

其實繼續打下去 只要手套不被薩諾斯拿走

感覺復仇者方應該還是能贏

應該只是要給鋼鐵人一個豪華的便當才這樣的吧?

後面就是各英雄回歸自己圓滿的結局 (東尼QQ

然後隊長要歸還無限寶石 以免其他時空混亂

54321之後 想不到居然沒回來

看起來只有巴奇不緊張 然後回頭一個老人

是回到過去過平靜生活的隊長 然後將盾牌與美國隊長 交接給了獵鷹

隊長的那支舞 終於跳到了

三本柱之所以如此令人感動

終究是漫威十年來各個電影細膩的刻畫這三個腳色

美國隊長 - 美國隊長123  復仇者1234

鋼鐵人 - 鋼鐵人123 復仇者1234

雷神 - 雷神123 復仇者1234

三個人都用了七部電影 細細刻畫著這三個腳色

美隊從過去 到現在 從對抗納粹 到現在對抗不同的外星人 堅持著自己的正義

鋼鐵人從一個商人 變成了保衛地球 宇宙的超級英雄 堅持著守護自己與所愛的人

雷神從阿斯嘉的王子 接著失去一切 又重新站起 到最後重新出發

感謝漫威十年如此精采的電影



最後 終究有幾個小小問題 不知道是Bug還是我沒注意 不知道有沒有人看到的

1. 皮姆粒子與量子衣

   2019涅布拉其實只有一組來回的皮姆粒子

   2014涅布拉把皮姆粒子給薩諾斯後 應該就沒辦法回到2019?

   就像美隊跟鐵人 要飛去1970的意思一樣 因為那裏不只有空間魔方 還有皮姆粒子

   而且薩諾斯大軍也都沒有量子衣 這樣到底怎麼飛到未來的= =? (太空船比較高級XD?

   而且回到過去的飛去佛米爾星的太空船又是哪來的?

2. 無限手套

   鋼鐵人很輕鬆地就做出了無限手套 然後把寶石放上去

   浩克博士還嚇了他一下

   復3不是說只有矮人做的無限手套才能發揮寶石的能力嗎?

3. A寶石

   最後東尼A走寶石 應該是因為他手套上有動手腳來著?

   也太好拔了 應該也只是為了華麗便當作準備

4. 美國隊長的圓盾

   不是在大戰時被打壞了一半 怎麼傳給獵鷹的時候又好了?

   雖然我覺得美隊只是純粹圓夢 應該沒有要解釋這段的合理性了

突然想到回來補充

覺得4的薩諾斯相較比較扁平跟壞人

不像3的有自己的理想跟抱負

應該是因為要塑造一個打爆他也不可惜的魔王吧


--
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testOneSent(self):
        phraseg = Phraseg('''
        好看 真的好看

一種十年磨一劍的感覺

基本上是三本柱 美國隊長 鋼鐵人 雷神索爾的電影 XD

一開始鷹眼開開心心的教女兒射箭 就知道沒五分鐘就要剩粉了QQ

接著是鋼鐵人的部分 雖然跟預告一樣 也知道他不會就掛在太空船

只是還是擔心 直到驚奇隊長閃閃發光的把鋼鐵人閃醒XD

飛回地球 鋼鐵人與美隊的小爭執 我們真的一起輸了 有種揪心感

就算後來飛去找薩諾斯 再次上演手套爭奪戰

索爾這次飛快的砍掉手(復3沒人有刀來著XD

手套掉下來 只是沒了寶石 沒了寶石 沒了寶石 

當下跟著影中人一起錯愕了一下

薩老緩緩說出我用寶石滅了寶石

涅布拉還默默補充我爸不會說謊...

雷神很雷的就把薩諾斯的頭給剁了 想問甚麼也沒得問了= =(雖然是有呼應復3拉

接著就進入哀傷的五年後

美隊在開導大家

黑寡婦在當代理局長

驚奇隊長又飛去宇宙了 所以只回來打了一架

發現沒辦法逆轉現況之後就又飛走了 說好的逆轉無限呢-.-

然後美隊跟黑寡婦聊天 接著被老鼠救出來的蟻人跑來

開啟了這集最重要的量子空間回過去

然後去找了幸福美滿的東尼QQ

可以理解東尼不想幫忙 因為他的模型還沒做出來(誤

只是看著小蜘蛛的照片 讓他又爬起來努力的跑量子模型

另一邊美隊去找了可以維持浩克先生的班納

然後各種搞笑XD

蟻人各種變 有點擔心他回不來 哈哈

這時 東尼帶著金頭腦跟盾牌回來啦~~~

隊長還是要拿著圓盾比較有FU

火箭跟浩克先生去找了雷神 喔 是肥神索爾回來

索爾真的很懊悔 從媽媽救不到 爸爸在眼前消失 弟弟被薩諾斯掛了

朋友因為自己沒有給薩諾斯致命一擊而灰飛煙滅了一半

五年來不斷酗酒走不出來也是蠻合理的

只是這集三本柱最雷的就是他了XD

然後黑寡婦去找到了變成浪人的鷹眼 把他帶回來

回到總部 分隊完

穿好量子戰衣

出發

美國隊長 + 鋼鐵人 + 蟻人 + 浩克先生-> 復1的紐約

火箭浣熊 + 肥神 -> 雷神2的阿斯嘉

涅布拉 + 戰爭機器 + 鷹眼 + 黑寡婦 -> 星際異工隊1

總覺得最後一個分組怪怪的 怎麼會給三個人類去外星球搶東西= =





復1的紐約

浩克看到自己以前的樣子 學著砸了一些東西的場景真的很逗趣XD

然後他去找當時守護寶石的古一 開始了一串辯論

浩克先生這集真的沒甚麼武戲 都在講話

然後隊長走進電梯 場景跟美2一模一樣

本來以為要開扁了 結果居然正直的美隊以一句Hail Hydra輕鬆A走了權杖

順便致敬漫畫 XD

也順勢解釋為什麼復1之後權杖又跑去九頭蛇那 做實驗做出了緋紅女巫&快銀

鋼鐵人&蟻人 本來要搶到空間魔方 結果來一個浩克討厭走樓梯XD

寶石就這樣被洛基A走順便烙跑

然後隊長走著走著遇到了自己

以為未來隊長是洛基的過去隊長 開始對A

然後又來一句隊長經典台詞

I can do this all day , 我知道XD

終究是自己了解自己 以一句巴奇還活著影響自己 順利尻飛過去隊長

然後權杖戳下去 帶走權杖

會合後再度前往1970 去找還在神盾局的魔方&皮姆

這邊偷東西就沒甚麼意外 再有意外就演不完了XD

這邊的主軸 給了鋼鐵人與爸爸的相會

為人父的鋼鐵人有太多事想跟爸爸講 終於圓夢

以及隊長看卡特 為結局埋了伏筆

原來東尼本來不叫東尼XD

阿斯嘉這邊

肥神各種雷 回到過去還在找酒

然後看到媽媽就忍不住想要回去找媽媽

完全不顧大局 單就這個行為真的很雷

只是這集就是給三本柱圓夢的 當然要給肥神回去找媽媽聊天

搶寶石就交給火箭 他也很輕鬆地搶到了寶石

然後圓完雷神的夢

雷神把喵喵槌叫了過來 帶回未來(當下OS是他把槌子拿走了 待會過去的雷神要怎麼用啊

不過後面就知道雷神回到過去最重要的其實就是把槌子拿走 哈哈哈哈





星際異攻隊這邊

鷹眼跟黑寡婦搭著太空船飛到了佛米爾星

話說 哪來的太空船

而且給兩個人類四處太空旅行 每個星球都有氧氣來著= =??

然後又遇到了守門員紅骷顱

接著開始誰要跳 鷹眼你射箭阻止黑寡婦的時候壓根就想炸死他吧XDDD

然後兩個人一起跳下去的時候

我以為會出現老梗之 互相珍惜所愛的人 並且能為對方而死

才是得到靈魂寶石的正確方法

結果是出現另一個老梗 下面的人鬆手往下跳

黑寡婦真的死了!? 靈魂寶石GET (當下想說不是有個人電影 後來才知道是前傳

涅布拉跟戰爭機器

一拳KO星爵就拿到寶石了

只是本篇最雷的涅布拉不小心跟過去涅布拉連結在一起

然後就被抓了 各種劇透給2014薩諾斯

知道復仇者在過去偷寶石之後 決定跟著到未來

因為寶石已經有人幫忙收集好了

只是 皮姆粒子不是只有一個人來回

後來是怎麼讓2014涅布拉去2023 又可以開啟量子通道給薩諾斯大軍過來呢??

畫面回到2023 過去不管多久 未來其實只過了一秒

所以也能解釋涅布拉被換了也不知道 因為未來沒有比較晚回來



接著把寶石裝上手套 (說好的手套要由矮人做才能發揮威力勒~~~

關在小房間 準備彈指

為啥不把捏布拉也關進來 放他在外面開量子通道 嘖嘖

順利彈指 浩克先生差點就掛了

外面鳥開始飛 鷹眼的手機響了

蟻人開始開心的說我們做到了!

只是看電影的大家都知道 糟了

接著總部就被各種轟炸

薩諾斯坐在外面等著倖存的復仇者出來

然後 三本柱 帥氣登場

開始三英戰呂布

只是薩諾斯就算沒手套也是很強

肥神應該是因為變太肥了 感覺很弱

還被壓著差點被自己的風暴毀滅者戳死

這時

隊長

舉起喵喵槌拉!!!!!!!

超帥 不解釋 看得我都泛淚了

然後帥打一波 還有槌子+盾牌合體技

只是帥打一波之後又被屌虐一波回來

連盾牌都被打飛一半

還被嘴+大軍壓境

隊長渾身傷

只默默拉緊盾牌的帶子

再次地站了起來

雖千萬人吾往矣阿!!!!

超帥的 帥到我又泛淚了

這時絕望感來到最高

一句 Cap do you copy! On your left

然後火圈開始冒出

終於

全部人都回來了

Avengers assemble

接著開始各種秀

鋼鐵小辣椒 女武神天馬 黃蜂女 黑豹 奇異博士 小蜘蛛

手套接力賽其實只是秀大家技能的時候

隊長跟索爾交換武器 索爾還說你拿小的就好

葛魔拉怒踢星爵蛋蛋 涅布拉說你只有他跟樹可以選

然後緋紅女巫屌打2014薩諾斯 只是2014薩諾斯表示你誰啊XD

2014薩諾斯表示: 我啥事都沒做 干我屁事XD

被女巫打得受不了了 惱羞開戰艦砲轟地面

還在想這台煩人的戰艦怎麼辦 就有道紅光來解決一切

驚奇隊長終於飛回來了- -

都快打完了還不回來

打飛戰艦後

給驚奇隊長跟薩諾斯單挑一波

寶石原來還可以單拿起來轟人 只能說薩諾斯真的很會用寶石

接著就是東尼看了正在治水的奇異博士

博士只默默伸出食指

然後東尼就衝去搶手套了

本來以為手套又搶失敗了

薩諾斯拉好手套 這次沒人阻止他了

薩諾斯: I'm Inevitable

!? 啥事都沒發生

只見東尼舉起手 寶石已經全都A過來了











                           東尼 : I am Iron Man











一彈指 一切就結束了

只是鋼鐵人 就這樣華麗的謝幕了

其實繼續打下去 只要手套不被薩諾斯拿走

感覺復仇者方應該還是能贏

應該只是要給鋼鐵人一個豪華的便當才這樣的吧?

後面就是各英雄回歸自己圓滿的結局 (東尼QQ

然後隊長要歸還無限寶石 以免其他時空混亂

54321之後 想不到居然沒回來

看起來只有巴奇不緊張 然後回頭一個老人

是回到過去過平靜生活的隊長 然後將盾牌與美國隊長 交接給了獵鷹

隊長的那支舞 終於跳到了

三本柱之所以如此令人感動

終究是漫威十年來各個電影細膩的刻畫這三個腳色

美國隊長 - 美國隊長123  復仇者1234

鋼鐵人 - 鋼鐵人123 復仇者1234

雷神 - 雷神123 復仇者1234

三個人都用了七部電影 細細刻畫著這三個腳色

美隊從過去 到現在 從對抗納粹 到現在對抗不同的外星人 堅持著自己的正義

鋼鐵人從一個商人 變成了保衛地球 宇宙的超級英雄 堅持著守護自己與所愛的人

雷神從阿斯嘉的王子 接著失去一切 又重新站起 到最後重新出發

感謝漫威十年如此精采的電影



最後 終究有幾個小小問題 不知道是Bug還是我沒注意 不知道有沒有人看到的

1. 皮姆粒子與量子衣

   2019涅布拉其實只有一組來回的皮姆粒子

   2014涅布拉把皮姆粒子給薩諾斯後 應該就沒辦法回到2019?

   就像美隊跟鐵人 要飛去1970的意思一樣 因為那裏不只有空間魔方 還有皮姆粒子

   而且薩諾斯大軍也都沒有量子衣 這樣到底怎麼飛到未來的= =? (太空船比較高級XD?

   而且回到過去的飛去佛米爾星的太空船又是哪來的?

2. 無限手套

   鋼鐵人很輕鬆地就做出了無限手套 然後把寶石放上去

   浩克博士還嚇了他一下

   復3不是說只有矮人做的無限手套才能發揮寶石的能力嗎?

3. A寶石

   最後東尼A走寶石 應該是因為他手套上有動手腳來著?

   也太好拔了 應該也只是為了華麗便當作準備

4. 美國隊長的圓盾

   不是在大戰時被打壞了一半 怎麼傳給獵鷹的時候又好了?

   雖然我覺得美隊只是純粹圓夢 應該沒有要解釋這段的合理性了

突然想到回來補充

覺得4的薩諾斯相較比較扁平跟壞人

不像3的有自己的理想跟抱負

應該是因為要塑造一個打爆他也不可惜的魔王吧


--
        ''')
        result = phraseg.extract(sent="覺得4的薩諾斯相較比較扁平跟壞人")
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testJapanese(self):
        phraseg = Phraseg('''
        作品の概要
        本作は、22世紀の未来からやってきたネコ型ロボット・ドラえもんと、勉強もスポーツも苦手な小学生・野比のび太が繰り広げる少し不思議（SF）な日常生活を描いた作品である。基本的には一話完結型の連載漫画であるが、一方でストーリー漫画形式となって日常生活を離れた冒険をするという映画版の原作でもある「大長編」シリーズもある。一話完結の基本的なプロットは、「ドラえもんがポケットから出す多種多様なひみつ道具（現代の技術では実現不可能な機能を持つ）で、のび太（以外の場合もある）の身にふりかかった災難を一時的に解決するが、道具を不適切に使い続けた結果、しっぺ返しを受ける」というものが多く、前作の「ウメ星デンカ」のストーリー構図をほぼそのまま踏襲しており実質的な後継作品ともいえる。このプロットは、作者の藤子・F・不二雄が自身のSF作品で描いた独自定義「すこし・不思議」（Sukoshi Fushigi）[注 2]という作風に由来し、当時の一般SF作品の唱える「if」（もしも） についての対象を想定した回答が反映されている。
        作品の主人公はドラえもんであるが、上記のプロットのように物語の主な視点人物はのび太である。
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testCantonese(self):
        phraseg = Phraseg('''
        「硬膠」原本是指硬身的塑膠，所以亦有「軟膠」，但在網上有人用此詞取其諧音代替戇鳩這個粵語粗口；直譯英語為 Hard Plastic，繼而引申出縮寫 HP。此用詞是諧音粗口，帶有不雅或恥笑成份。部分人不接受說「膠」字，認為膠是粗口之一[1]。
        硬膠亦簡稱「膠」，可作名詞、形容詞、動詞等使用。硬膠雖與「on9」的語調上不同，但意思差不多，有時可以相通。小丑icon和小丑神也是硬膠的形象化的圖像顯示和象徵。
        「硬膠」一詞聽聞歷史悠久，但出處不明，而由香港高登討論區將其發揚光大，現己推展至香港其他主要網絡社區。在2002年時，有些網民利用「戇鳩」的諧音，發明「硬膠」一詞，更把愛發表無厘頭帖子的會員腦魔二世定為「硬膠」始祖。自此，「硬膠文化」便慢慢發展起來，某程度上硬膠文化與無厘頭文化相似。因腦魔二世的「硬膠」功力驚人，更成為當時被剝削的會員的救星，可令他們苦中作樂一番，故有人曾經預言：「救高登，靠膠人」。及後，高登會員以縮寫「膠」來代替硬膠。而高登亦有了「膠登」的綽號。當時甚至出現了7位膠力驚人的高登會員，以腦魔二世為首，合稱為「硬膠七子」。
        其實「硬膠」早於周星馳電影《整蠱專家》早已出現雛型。戲中有一台詞為「超級戇膠膠」，可見膠字是用作取代粵語粗口鳩字。
        有網友提供資料指西方早於上世紀六十年代亦已經將「膠」（Plastics）等同愚蠢：
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testKorean(self):
        phraseg = Phraseg('''
        "서울"의 어원에 대해서는 여러 가지 설이 존재하나, 학계에서는 일반적으로 수도(首都)를 뜻하는 신라 계통의 고유어인 서라벌에서 유래했다는 설이 유력하게 받아들이고 있다. 이때 한자 가차 표기인 서라벌이 원래 어떤 의미였을지에 대해서도 여러 학설이 존재한다. 삼국사기 등에서 서라벌을 금성(金城)으로도 표기했다는 점과 신라(新羅)까지 포함하여 "설[새: 新, 金]-벌[땅: 羅, 城]", 즉 '새로운 땅'이라는 뜻으로 새기는 견해가 있다. 고대-중세 한국어에서 서라벌을 정확히 어떻게 발음했는지에 대해서는 확실하게 확인된 바가 없으며, 그 발음을 알 수 있게 되는 것은 훈민정음의 창제 후 "셔ᄫᅳᆯ"이라는 표기가 등장하고 나서부터이다.
        조선 시대에는 서울을 한양 이외에도 경도(京都), 경부(京府), 경사(京師), 경성(京城), 경조(京兆) 등으로 쓰는 경우가 종종 있었으며, 김정호의 수선전도에서 알 수 있듯 수선(首善)으로 표기한 예도 있다. 그 밖의 표기 중에는 서울의 한자 음차 표기로서 박제가가 북학의에서 썼던 '徐蔚(서울)'이 있다. 이는 모두 수도를 뜻하는 일반명사들로서 '서울'이 원래는 서울 지역(사대문 안과 강북의 성저십리)을 가리키는 말이 아닌 수도를 뜻하는 일반명사였다는 방증이다. 국어사전에서는 일반명사 '서울'을 '한 나라의 중앙 정부가 있고, 경제, 문화, 정치 등에서 가장 중심이 되는 도시'라고 정의하고 있다.[4] 1910년 10월 1일에 일제가 한성부를 경성부(京城府)로 개칭하면서 일제강점기에 서울은 주로 경성(京城, 일본어로는 けいじょう)으로 불렸으며, 1945년 광복 후에는 '경성'이란 말은 도태되고 거의 '서울'로 부르게 되었다.[
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testFrance(self):
        phraseg = Phraseg('''
        El idioma francés (le français /lə fʁɑ̃sɛ/ ( escuchar) o la langue française /la lɑ̃ɡ fʁɑ̃sɛz/) es una lengua romance hablada en la Unión Europea, especialmente en Francia, país en el que se habla junto con otras lenguas regionales como el idioma bretón (Bretaña), el occitano (Occitania), el vasco (país vasco francés), el catalán (Rosellón), y el corso (Córcega). En los territorios franceses de ultramar es hablado en muchos casos junto con otras lenguas como el tahitiano (Polinesia Francesa), o el créole (isla Reunión, Guadalupe y Martinica). También se habla en Canadá, Estados Unidos (francés cajún, créole y francés acadio o acadiano), Haití (con el créole), y numerosos países del mundo. Según estimaciones de la Organización Internacional de la Francofonía (basadas en proyecciones demográficas de las Naciones Unidas), en el transcurso del s. XXI, el francés se convertiría en el tercer idioma con el mayor número de hablantes del mundo, sobre todo por el crecimiento poblacional de los países africanos francófonos.5​
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testSpanish(self):
        phraseg = Phraseg('''
        Breaking Bad es una serie de televisión dramática estadounidense creada y producida por Vince Gilligan. Breaking Bad narra la historia de Walter White (Bryan Cranston), un profesor de química con problemas económicos a quien le diagnostican un cáncer de pulmón inoperable. Para pagar su tratamiento y asegurar el futuro económico de su familia comienza a cocinar y vender metanfetamina,1​ junto con Jesse Pinkman (Aaron Paul), un antiguo alumno suyo. La serie, ambientada y producida en Albuquerque (Nuevo México), se caracteriza por poner a sus personajes en situaciones que aparentemente no tienen salida, lo que llevó a que su creador la describa como un wéstern contemporáneo.2​
        La serie se estrenó el 20 de enero de 2008 y es una producción de Sony Pictures Television. En Estados Unidos y Canadá se emitió por la cadena AMC.3​ La temporada final se dividió en dos partes de ocho episodios cada una y se emitió en el transcurso de dos años: la primera mitad se estrenó el 15 de julio de 2012 y concluyó el 2 de septiembre de 2012, mientras que la segunda mitad se estrenó el 11 de agosto de 2013 y concluyó el 29 de septiembre del mismo año.
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testThai(self):
        phraseg = Phraseg('''
        กิตฮับ (อังกฤษ: GitHub) เป็นเว็บบริการพื้นที่ทางอินเทอร์เน็ต (hosting service) สำหรับเก็บการควบคุมการปรับปรุงแก้ไข (version control) โดยใช้กิต (Git) โดยมากจะใช้จัดเก็บรหัสต้นฉบับ (source code) แต่ยังคงคุณสมบัติเดิมของกิตไว้ อย่างการให้สิทธิ์ควบคุมและปรับปรุงแก้ไข (distributed version control) และระบบการจัดการรหัสต้นฉบับรวมถึงทางกิตฮับได้เพิ่มเติมคุณสมบัติอื่นๆผนวกไว้ด้วย เช่น การควบคุมการเข้าถึงรหัสต้นฉบับ (access control) และ คุณสมบัติด้านความร่วมมือเช่น ติดตามข้อบกพร่อง (bug tracking), การร้องขอให้เพิ่มคุณสมบัติอื่นๆ (feature requests), ระบบจัดการงาน (task management) และวิกิสำหรับทุกโครงการ[2]
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) > 1)
        self.assertTrue(len(result) > 1)

    def testLang(self):
        phraseg = Phraseg('''
        ''')
        result = phraseg.extract()
        print(result)
        self.assertTrue(len(phraseg.ngrams) == 0)
        self.assertTrue(len(result) == 0)


if __name__ == '__main__':
    unittest.Test()

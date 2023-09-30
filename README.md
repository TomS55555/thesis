# Thesis
The code for my thesis: EEG-based sleep-staging with SSL, read the full text: #https://github.com/TomS55555/thesis/edit/master/Thesis_final_pdf.pdf

# Abstract
This thesis aims to reduce the reliance on labeled data for training a machine learning
algorithm for EEG-based sleep staging. To achieve this, we propose leveraging
unlabeled data through self-supervised learning (SSL) methods. These methods aim
to create proxy tasks that learn information about the data without having any
task-related labels. Obtaining labeled datasets for biomedical applications, such as
sleep staging, can be challenging. However, unlabeled data is often more abundant
and accessible. Thus, exploring SSL methods becomes valuable in this context.

The current state-of-the-art deep learning architectures for sleep staging are
sequence-to-sequence models composed of an epoch encoder, a sequence encoder, and
a classifier. First, an epoch encoder extracts features from the raw EEG data. Then,
a sequence encoder augments these features with temporal dependencies and encodes
information like stage transition rules. Finally, a classifier classifies a sequence of
encoded input EEG epochs into the respective sleep stages.

In this thesis, a two-step SSL paradigm is proposed for sequence-to-sequence sleep
staging architectures. Specifically, the proposal suggests pretraining the epoch encoder
and the sequence encoder with different SSL objectives suitable for feature extraction
and sequence modeling, respectively. Similar approaches have been successful in
natural language programming (NLP) with models like GPT [8]. However, it has not
been explored for EEG-based sleep staging, and this thesis demonstrates its viability.

The epoch encoder is pretrained using SimCLR [10], a contrastive framework
that has shown success in a wide variety of applications. We show that the feature
space obtained after pretraining with SimCLR effectively clusters the sleep stages.
Additionally, a classifier trained on top of this pretrained epoch encoder outperforms
the regular supervised model by 5 percentage points when only a small fraction
(approximately 1%) of the labeled dataset is used for both.

In the second phase, a pretext task is designed to further pretrain the sequence
encoder on top of the pretrained epoch encoder. This allows the sequence encoder to
learn context from neighboring epochs. We demonstrate that the pretrained sequence
encoder also outperforms its supervised counterpart, albeit by a smaller margin of 3
percentage points in the low data regime.

Overall, our two-stage SSL pretraining approach improves the sleep staging
accuracy compared to random weight initialization. It achieves a 10% improvement
in the low-data regime (using approximately 1% of the labeled dataset) and a 7%
improvement when using 10% of the labeled dataset.

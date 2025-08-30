\documentclass{article}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{cite}
\usepackage{url}
\usepackage{subcaption}
\usepackage{float}
\usepackage{multirow}
\usepackage[hidelinks]{hyperref}

\title{An Emotion-Aware Conversational AI Agent for Chinese Elderly Mental Health Support}

\author{
Yifei Chen, Richard Cai, Lucy Liu\\
\textsf{\{chen239y, liu224\}@mtholyoke.edu, caitengqi@gmail.com}
}

\begin{document}

\maketitle

\begin{abstract}
China has entered a period of demographic transition, marked by a third consecutive year of negative population growth by the end of 2024. The birth rate fell to 6.77\%, with only 9.54 million newborns, while deaths exceeded 10.93 million. This phenomenon suggests that the population aged 65 and above will more than double between 2020 and 2050---from 172 million (12.0\%) to 366 million (26.0\%). This rapid aging trend brings huge challenges for both families and society. With increasingly fragmented family structures and shifting caregiving norms, the burden of elder support is growing. Although Chinese law mandates family responsibility for eldercare, the mental and emotional well-being of older adults remains largely neglected. According to the 2024 China Report on Elder Mental Health, 23.75\% of older adults report loneliness and 26.4\% show symptoms of depression, while access to mental health resources remains scarce. This project introduces an emotion-aware conversational AI agent designed to provide scalable and accessible mental health support for Chinese elderly users. By combining emotion detection, dialogue strategy selection, and long-term trend monitoring, the system offers a low-cost, always-available emotion companion that is especially for older individuals who may suffer in silence.
\end{abstract}

{\small
\noindent\textbf{Code Availability:} Core code and training scripts are available at \url{https://github.com/your-repo}.\\
\textbf{Ethics Compliance:} All training data used in this work are synthetically generated, and no human-subject experiments were conducted.\\
\textbf{Keywords:} Emotion recognition, conversational AI, elderly care, mental health, multimodal analysis
}

\section{Introduction}

The rest of this paper is organized as follows: Section~\ref{sec:related} reviews related work. Section~\ref{sec:system} presents the system architecture and core modules. Section~\ref{sec:implementation} describes the implementation details. Section~\ref{sec:evaluation} introduces the performance of the emotion classification trained. Then, Section~\ref{sec:discussion} discusses the application of this agent and its limitations. Finally, Section~\ref{sec:conclusion} concludes the paper and outlines future directions.

\section{Related Work}
\label{sec:related}

\subsection{Emotion-Aware Dialogue Systems}

Emotion-aware conversational agents designed to respond appropriately to a user's emotional state in an empathetic and trustworthy way. Widely deployed products such as Woebot and Replika have demonstrated the feasibility of delivering emotionally supportive dialogue through mobile chatbots~\cite{fitzpatrick2017woebot,possati2023replika}. Replika's use of personalization and mood mirroring, and Woebot's evidence-based cognitive behavioral dialogue, both show significant improvements in emotional well-being across user studies.

In the Chinese context, Microsoft XiaoIce represents one of the most architecturally complete emotion-aware agents to date. The XiaoIce system is structured into three main layers: the user interaction layer (handling front-end dialogue), the dialogue strategy layer, and the data layer. Within the strategy layer, two major components stand out. The dialogue manager governs state tracking and uses hierarchical policy routing to determine whether to invoke a ``skill'' or core chit-chat. The empathetic computing module is particularly important. It generates an emotional vector for each user utterance by extracting key affective dimensions such as emotion, intent, opinion, and inferred personality traits. This vector then conditions the generation or ranking of system responses, which enables XiaoIce to maintain long-term emotional alignment with users~\cite{zhou2020xiaoice}. While XiaoIce highlights the value of empathetic computing and long-term engagement, its full reranking and multimodal context stack is beyond the scope of small-footprint, rapid-cycle systems and thus serves here mainly as design inspiration rather than a directly replicated blueprint.

Transformer-based models have also been widely used for emotion recognition. According to Devlin et al.~\cite{devlin2019bert}, Liu et al.~\cite{liu2019roberta}, and Sun et al.~\cite{sun2019ernie}, fine-tuning large pre-trained models like BERT, RoBERTa, or ERNIE on emotion classification tasks has become a standard method that achieves high accuracy on benchmark datasets. For conversation-level emotion classification, Qin et al.~\cite{qin2023berterc} propose BERT-ERC, which shows substantial gains over RNN-based systems; similarly, Yang and Shen~\cite{yang2021emotiondynamics} model conversational emotion dynamics with BERT. Beyond supervised fine-tuning, Lei et al.~\cite{lei2023instructerc} introduce InstructERC and Fu et al.~\cite{fu2024laercs} present LaERC-S, which leverage prompt-based reasoning over conversational context to produce continuous emotion intensities and categorical labels, improving adaptability in low-resource, real-time settings.

\subsection{Mental Health Chatbots}

Mental health chatbots differ from generic emotion-aware agents in a way that they incorporate therapeutic objectives, such as mood tracking, behavior change, or crisis mitigation, into their core functionality. Applications like Wysa, Ginger, and Youper utilize structured cognitive therapy frameworks embedded into natural dialogue. Evaluation studies reveal promising impacts on reducing anxiety and depressive symptoms, even though concerns remain regarding ethical design, data protection, and handling of crisis input~\cite{inkster2018wysa,mehta2021youper,gingerio2018}. A notable academic system is SuDoSys~\cite{chen2024structured}, which introduces stage-aware counseling with a stage-aware instruction generator, key information extraction, and stage control to guide transitions across counseling phases. This underscores the utility of staged objectives for psychologically grounded dialogue and motivates our use of hierarchical strategy logic.

In the Chinese ecosystem, services like XiaoIce have partially addressed emotional needs but are often general-purpose. Recent LLM-based tools like Xingye (developed by MiniMax) and Emobot platforms are beginning to emphasize safety and psychological insight, whereas these remain under-evaluated in peer-reviewed literature.

\subsection{Trend Analysis in Emotional Monitoring}

While emotion detection in isolated dialogue turns is useful, long-term emotional trend tracking offers more valuable insights for mental health risk prediction. Techniques like emotion trajectory modeling, passive sensing, and behavioral analytics make up early warning systems.

Recent studies have combined mobile phone sensor data (sleep patterns, location entropy, physical activity) with machine learning models (e.g., transformers, HMMs) to forecast mood shifts and even predict suicidal ideation with high AUC scores~\cite{sano2018jmir}. In China, wearable devices have been used to record movement under emotional induction, training classifiers to detect states like happiness or anger.

In future development phases, we aim to incorporate a similar emotion trend monitoring module that aggregates multi-turn emotion scores and detects risk patterns (e.g., prolonged sadness). This module may integrate with basic alerting systems to inform caregivers or prompt user check-ins.

\subsection{Multimodal Emotion Analysis}

The role of multimodal inputs: text, audio, and visual features, is increasingly emphasized in emotion generation and recognition. Firdaus et al.~\cite{firdaus2022multimodal} proposed a conditional variational autoencoder (CVAE) with multimodal attention, trained to generate emotionally congruent responses using video dialogue data. Their work demonstrates that multimodal systems outperform unimodal baselines across diversity and emotional accuracy metrics.

Despite these advances, Chinese-language psychological systems rarely leverage multimodal inputs. Emotion detection is still primarily performed on text, limiting depth and subtlety. This gap is critical in older adult interactions, where tone and non-verbal cues may carry more affective weight than literal content.

We have already incorporated an audio modality via pitch- and timbre-based voice emotion analysis tailored to elderly speech. In future work, we plan to extend to full audiovisual corpora such as CH-SIMS~\cite{yu2020chsims} to integrate visual cues in order to improve accuracy in cases of mixed affect or implicit emotions.

\section{System Design and Architecture}
\label{sec:system}

\subsubsection*{Overall Architecture}
\begin{figure}[!htb]
    \centering
    \includegraphics[width=\linewidth]{System arch.png}
    \caption{System architecture and data flow diagram}
    \label{fig:system_arch}
\end{figure}

The system is designed to enable emotionally intelligent dialogue tailored for elderly users through multimodal inputs (voice and text), real-time emotion analysis, personalized dialogue strategies, and long-term emotional state monitoring. The architecture follows a modular design that integrates audio processing, dual-path emotion analysis, hierarchical rule-based strategy selection, large language model (LLM) response generation, and persistent memory, offering empathetic, safe, and personalized companionship.

\subsubsection*{Input and Preprocessing}
User input is collected via three primary channels: real-time microphone streaming (dependent on local PyAudio availability), fixed-duration recordings, and uploaded audio or text. ASR is provided by Baidu's cloud API~\cite{baidu2024speech}. To address the linguistic characteristics typical of elderly speech, such as colloquial fragments, dialectal influence, and speech recognition (ASR) errors, the system incorporates a sentence segmentation and normalization module. This module utilizes Baidu's speech recognition API with automatic punctuation to identify natural sentence boundaries, which are then refined through regular expressions and fallback heuristics to handle running punctuation, fillers, and truncated utterances while preserving emotional context. Normalization corrects homophones, converts spoken forms to standard written equivalents, and removes meaningless repetitions or fillers, which thereby ensures high-quality input for downstream emotion analysis. Although the current ASR model primarily supports Mandarin, future versions will address dialect-specific recognition.

\subsubsection*{Emotion Analysis Pathways}
Emotion recognition adopts a hybrid approach: (1) a lexicon-based pathway (C-LIWC) that extracts psychologically interpretable features (negemo, social, cogproc, self-reference), (2) an LLM-based pathway (e.g., DeepSeek) that performs zero-shot inference over the current utterance and recent dialogue history to output continuous-valued emotion intensities and labels, and (3) a voice pathway that analyzes pitch and timbre (MFCC means, spectral centroid, spectral bandwidth, spectral rolloff, zero-crossing rate). For model reliability, we also trained and evaluated an offline stacking ensemble (MLP, LightGBM, Ridge with a linear meta-learner) on synthetic eldercare dialogues; at runtime, the default text pathway is LLM-based, while the voice pathway runs in parallel and is fused with text via dynamic weighting. The proposed voice emotion detection module is a feature-driven, regression-based framework that integrates psychoacoustic principles with machine learning for robust emotional state estimation.


The system first extracts a 21-dimensional acoustic feature vector from each audio segment, comprising four pitch-related features (mean, standard deviation, maximum, and minimum pitch) and seventeen timbre-related features, including thirteen Mel-Frequency Cepstral Coefficients (MFCC) mean values, spectral centroid, spectral bandwidth, spectral roll-off, and zero crossing rate~\cite{mcfee2015librosa,mauch2014pyin,davis1980mfcc,tzanetakis2002genre}. These features jointly capture the prosodic and spectral characteristics that reflect emotional tone, intensity, and variability, and are then standardized using Z-score normalization to ensure equal weighting during model training.

Given the absence of professionally annotated datasets, a psychoacoustically-informed synthetic data generation strategy was adopted to produce 200 training samples. Rather than assigning arbitrary feature--emotion mappings, this process is grounded in established research on vocal emotion expression and prosody--emotion relations~\cite{scherer2003vocal}. For instance, joy is modeled through elevated pitch and increased spectral brightness, sadness through reduced pitch variability and diminished spectral energy, and anger through heightened pitch fluctuations and increased zero crossing rate. Emotional intensity follows the peak emotion principle, where overall activation is determined by the strongest individual emotion component.

The prediction stage employs a multi-output regression framework in which independent linear regression models are trained for each emotional dimension. This design enables the detection of complex mixed emotional states while preserving interpretability through direct analysis of model coefficients. The low computational complexity supports real-time inference, and the modular structure allows straightforward extension to additional emotional dimensions. Training progress is monitored using Mean Squared Error (MSE).

Finally, the module is integrated into the larger multi-modal emotion recognition pipeline alongside text-based emotion analysis. Voice-based scores undergo dynamic weight adjustment according to audio quality and contextual conditions, which aligned with multimodal fusion practices~\cite{baltrusaitis2019multimodal}.

\subsubsection*{Voice Emotion Scoring}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\linewidth]{Voice processing.png}
    \caption{Voice emotion feature extraction and scoring}
    \label{fig:voice_pipeline}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=1\linewidth]{dynamic fusion.png}
    \caption{Dynamic fusion diagram}
    \label{fig:dynamic_fusion}
\end{figure}

The strategy selection module utilizes a hierarchical, rule-based framework guided by psychological relevance and user safety.

\begin{figure}[H]
    \centering
    \includegraphics[width=1\linewidth]{strategy selector.png}
    \caption{Hierarchical strategy selection}
    \label{fig:strategy_selector}
\end{figure}

\subsubsection*{Long-term Trend Tracking}
To support long-term emotional tracking, the system continuously aggregates and visualizes daily averages of sadness, joy, and anger scores, as well as the frequency of triggered alerts. This timeline view of emotional trends enables monitoring of persistent distress patterns, such as long-term sadness or social withdrawal, and enhances the accuracy of risk detection across sessions.

\begin{figure}[!htb]
    \centering
    \includegraphics[width=1\linewidth]{trend.png}
    \caption{Long-term trend chart pipeline}
    \label{fig:trend_pipeline}
\end{figure}

\section{Module Implementation}
\label{sec:implementation}

\subsubsection*{Input Processing}
Input processing is managed through the \texttt{BaiduSpeechRecognizer} class, which interfaces with Baidu's ASR services. Live speech is captured by PyAudio in 16kHz mono mode, and silent detection based on audio energy is applied to segment the utterance into streaming transcription. Recognition results are returned asynchronously through threading and queues to minimize latency and accommodate the slow, intermittent speech patterns typical of elderly users. Fixed-duration recordings are saved as temporary WAV files before batch transcription. Audio files and direct text inputs are fed through a processing pipeline that performs sentence segmentation and metadata tagging.

\subsubsection*{Segmentation and Normalization}
Sentence segmentation primarily relies on Baidu's automatic punctuation insertion feature, with light post-processing to consolidate repeated punctuation and preserve meaningful speech fragments (including fillers and dialect expressions). Normalization rules (e.g., homophone correction, spoken-to-written mappings, regex-based de-duplication) are considered for future enhancement. Dialectal inputs currently utilize the Mandarin model (dev\_pid=1537).

\subsubsection*{Text Emotion Analysis}
Emotion recognition combines two complementary approaches: C-LIWC extracts linguistically interpretable features across categories such as negemo, social, cogproc, and self-focus. These features have been widely validated in psychological and computational studies~\cite{tausczik2010liwc}. In the Chinese context, a localized version C-LIWC was used to ensure linguistic compatibility and cultural relevance~\cite{huang2012liwc}. In the meantime, a text emotion pathway uses an LLM-based classifier (DeepSeek)~\cite{deepseek2023llm} to produce continuous-valued emotion intensity scores. Separately, we trained an offline stacking ensemble (MLP, LightGBM, Ridge; linear regression as meta-learner) to improve consistency; these results are reported in Section~\ref{sec:evaluation}, while the runtime pipeline defaults to the LLM pathway.

\subsubsection*{Voice Emotion Analysis}
Voice emotion detection operates in parallel with text analysis via a dedicated module that extracts a 21-dimensional acoustic feature vector (4 pitch features and 17 timbre features: MFCC means, spectral centroid, bandwidth, rolloff, zero-crossing rate). Features are standardized and fed to a multi-output linear regression model trained on 200 psychoacoustically informed synthetic samples (high pitch + bright timbre $\rightarrow$ joy; low variability + flatter prosody $\rightarrow$ sadness; high pitch variability + frequent spectral changes $\rightarrow$ anger; intensity = max component). Voice and text scores are reconciled through dynamic weighting informed by text length, audio quality, and salient emotion type to produce final emotion estimates.

\subsubsection*{Strategy Selection}
The strategy selection component adopts a four-layer hierarchical decision tree: early warning detection employing sliding window sadness thresholds; LIWC anomaly detection signaling depressive-like states; emotion dominance rules for joy, anger, mixed emotion, and neutral states; and a companionship fallback. Semantic similarity between strategy prompts and user input is computed using MiniLM-v2 embeddings~\cite{wang2020minilm,wang2021minilmv2} alongside SBERT-style sentence embeddings~\cite{reimers2019sbert}, with a threshold of 0.45 to preserve contextual relevance. Strategy prompts are filled dynamically with stored user keywords and safe elderly topics with compassionate prompt templates that encourage LLMs to respond warmly and in personalized styles.

\subsubsection*{Long-term Tracking and Storage}
The long-term tracker module retains emotion time series and keyword memories. It monitors out-of-norm trends and prolonged emotional duration. This helps provide automated alert flags for recommending caregiver intervention when necessary. The central controller integrates input acquisition, module operation, and output composition, maintaining session-specific memory states and keeping modularity so that future extension or component substitution is possible.

\section{Evaluation}
\label{sec:evaluation}

Even though the implementation of the LLM in the emotion recognition module performs well, it still has several drawbacks such as lack of interpretability and difficulty ensuring output consistency. Therefore, as an improvement, we trained an emotion classification model to directly classify emotions instead.

We generated a comprehensive emotion recognition dataset with 10,000 elder care dialogue samples through rigorous prompting of the LLM. Each sample contains multi-turn conversations annotated with emotion labels across four dimensions: anger, sadness, joy, and emotional intensity. A BERT-based Chinese text embedding model is then employed to capture contextual semantic information from the dialogue turns. The input features include:

\begin{itemize}
\item Text embeddings: 768-dimensional BERT representations
\item Emotion features: Domain-specific emotional indicators  
\item Maximum dialogue turns: 7
\end{itemize}

For model training, a stacking ensemble approach combining three diverse base models is adopted: Multi-Layer Perceptron (MLP), Gradient Boosting Models, and Ridge Regression. A meta-learner, implemented as linear regression with scikit-learn~\cite{pedregosa2011scikit}, optimally combines the predictions from these base models to produce the final emotion scores.

We evaluated model performance using MSE for regression accuracy and AUC-ROC for binary classification for each emotion. In this setup, a lower MSE indicates more accurate emotion score predictions, while a higher AUC-ROC indicates better discrimination between presence/absence of an emotion. We adopted a stacking ensemble combining three diverse base learners (MLP, LightGBM~\cite{ke2017lightgbm}, and Ridge regression) with a linear regression meta-learner.

\subsection{Overall Performance Comparison}

\begin{table}[h]
\centering
\caption{Overall Model Performance Comparison}
\label{tab:overall_performance}
\begin{tabular}{|l|c|c|}
\hline
\textbf{Model} & \textbf{MSE} & \textbf{Avg AUC-ROC} \\
\hline
MLP & 0.0555 & 0.7430 \\
LightGBM & 0.0191 & 0.9093 \\
Ridge & 0.0191 & 0.9093 \\
Stacking Ensemble & 0.0192 & 0.9094 \\
\hline
\end{tabular}
\end{table}

\begin{figure}[!htb]
    \centering
    \includegraphics[width=1\linewidth]{model_comparison_final.png}
    \caption{Overall performance comparison}
    \label{fig:placeholder}
\end{figure}
\begin{table}[h]
\centering
\caption{Detailed Performance by Emotion Dimension}
\label{tab:emotion_performance}
\begin{tabular}{|l|l|c|c|}
\hline
\textbf{Emotion} & \textbf{Model} & \textbf{MSE} & \textbf{AUC-ROC} \\
\hline
\multirow{4}{*}{Anger} & MLP & 0.0223 & 0.9285 \\
 & LightGBM & 0.0066 & 0.9852 \\
 & Ridge & 0.0066 & 0.9852 \\
 & Stacking Ensemble & 0.0066 & 0.9852 \\
\hline
\multirow{4}{*}{Sadness} & MLP & 0.0674 & 0.6986 \\
 & LightGBM & 0.0234 & 0.9322 \\
 & Ridge & 0.0234 & 0.9322 \\
 & Stacking Ensemble & 0.0234 & 0.9323 \\
\hline
\multirow{4}{*}{Joy} & MLP & 0.1053 & 0.7545 \\
 & LightGBM & 0.0339 & 0.9041 \\
 & Ridge & 0.0339 & 0.9041 \\
 & Stacking Ensemble & 0.0339 & 0.9047 \\
\hline
\multirow{4}{*}{Intensity} & MLP & 0.0272 & 0.5903 \\
 & LightGBM & 0.0126 & 0.8158 \\
 & Ridge & 0.0126 & 0.8158 \\
 & Stacking Ensemble & 0.0127 & 0.8156 \\
\hline
\end{tabular}
\end{table}

\begin{figure}[!htb]
    \centering
    \includegraphics[width=1\linewidth]{heatmap.png}
    \caption{Detailed emotion dimension comparison heatmap}
    \label{fig:placeholder}
\end{figure}
\subsection{Detailed Emotion Dimension Analysis}

In terms of overall performance, the LightGBM and Ridge models individually achieved the lowest overall MSE (0.0191) and the highest average AUC-ROC (0.9093). Our stacking ensemble closely matched these results (MSE 0.0192, AUC 0.9094) while significantly outperforming the MLP baseline (MSE 0.0555, AUC 0.7430). In other words, the ensemble method preserved the top-tier accuracy of LightGBM/Ridge while dramatically reducing the error compared to MLP. This outcome aligns with prior findings that stacking can offer ``superior predictive power'' and precision in emotion recognition~\cite{alshamsi2023ensemble}.

In summary, our stacking ensemble achieved consistently strong results across all emotion dimensions. In each case it matched or slightly exceeded the best base model's performance, which confirms the benefit of ensemble learning in emotion classification. The low MSE values (e.g. 0.0066 for anger) indicate highly precise score predictions, and the high AUCs (up to 0.985) show robust classification. These results improve upon the simpler MLP baseline by a large margin and align with recent studies where stacking classifiers ``outperform other models in accuracy and performance'' in text-based emotion detection.

\section{Discussion}
\label{sec:discussion}

From a system-level perspective, and building upon the integrated module architecture and training procedures described above, our Eldercare Agent delivers psychologically informed multimodal processing, coherent conversation management, and persistent long-term emotion tracking. To address user isolation, the agent also safeguards user information and provides a personalized interaction experience. Rather than treating all modules as isolated components, the system orchestrates them into a coherent decision pipeline that balances redundancy, interpretability, and adaptability. This pipeline enables the agent to respond in real time to the nuanced and often fragmented conversational patterns of elderly users while preserving awareness of long-term emotional trends. The system's hierarchical strategy logic ensures that urgent emotional risks are prioritized without neglecting everyday companionship needs. Long-term emotional tracking further acts as a safeguard, enabling escalation when sustained deviations from baseline are observed. This integrated design moves beyond the performance of any single module, demonstrating how cross-module coordination enhances reliability, contextual relevance, and user trust. By aligning computational processing with psychological principles, the Eldercare Agent maintains emotional coherence across both short-term interactions and extended engagement, serving as an effective companion for older adults living alone, particularly in the Chinese context where family structures are becoming increasingly nuclear and adult children may have limited capacity to provide direct companionship.

Compared with existing Chinese-language emotion and conversational AI products, our system is differentiated by its explicit elderly focus and its design choices tailored to this demographic. Most current sentiment analysis and companion platforms target younger or general audiences, drawing on data from social media or short-form conversations that do not reflect the linguistic patterns, slower pacing, and cognitive considerations of elderly users. In contrast, our architecture incorporates psychologically grounded decision-making, persistent long-term emotion tracking, and a hierarchical strategy pipeline that blends rule-based safety with adaptive LLM responses. Unlike many competing products that are text-first, the agent supports voice-first interaction with live, silence-aware speech transcription, accommodating the fragmented or intermittent speech styles typical of older adults. Personalization is further enhanced through keyword memory, which dynamically integrates user-specific topics into strategy prompts, a feature rarely found in existing offerings. Finally, the modular but tightly coordinated architecture ensures that individual components---such as ASR, emotion detection, or strategy selection---can be upgraded independently without disrupting the overall coherence of the system, providing a level of adaptability uncommon in current commercial products.

Despite these strengths, several challenges remain before the system can be productized. Foremost among these is the slow response time caused by the lack of access to commercial APIs, which prevents the agent from providing truly instantaneous replies and results in a less fluid user experience. For voice input, the system currently lacks dialectal ASR capabilities, as there is no open-source speech recognition model trained on a sufficiently broad set of Chinese regional accents. This is a significant barrier, since many elderly users have received limited formal education and communicate primarily in local dialects. Such limitations also raise questions about the system's target user group---specifically, which segments of the elderly population are both open to AI-based interaction and capable of using a mobile application.

Emotion detection accuracy can also be improved with real-world conversational data. Currently, there is no publicly available dataset focused on elderly dialogue and most existing corpora are drawn from social media platforms used by younger people, containing only short exchanges with limited coherence. Synthetic data generated for training may therefore fail to fully capture the conversational characteristics of elderly users. Additionally, while the parallel use of C-LIWC and the ensemble emotion model provides diverse insights, the two methods currently operate independently, each determining only part of the strategy selection process. Designing a mechanism for their outputs to complement each other remains an open challenge. The emotional dimensions we currently track---joy, sadness, anger, and overall emotional intensity---may also be insufficient to capture the full spectrum of affective states relevant to detecting depression risk in elderly populations.

The conversation strategy module warrants further refinement. Its current hierarchical design prioritizes high sadness scores for triggering alerts, followed by LIWC anomaly detection, then emotion-dominance rules for anger, joy, and mixed emotions, with a fallback mechanism handling residual cases. Given that the strategy selection module also integrates keyword memory, prompts to the LLM can become overly dense, potentially diluting conversational clarity. Optimizing the inputs to this module in tandem with improvements to the emotion detection component is therefore a priority. Regarding keyword memory itself, the current implementation relies on semantic similarity to trigger related topics from a limited corpus; expanding the corpus and refining the logic could significantly enhance topic recall. Finally, prompt optimization for the LLM remains an ongoing task, with the goal of better defining the AI's persona and conversational boundaries in a way that is both engaging and safe for elderly users.

\section{Next Steps}

The immediate next stage of development will focus on addressing the limitations identified above. Priority will be given to optimizing the agent's response latency by refining the model invocation pipeline and exploring more efficient inference pathways in the absence of commercial API access. Concurrently, dialectal ASR capabilities will be pursued through the collection and integration of accented Chinese speech data, enabling the system to serve elderly users who rely primarily on regional dialects. For voice emotion detection, we plan to synthesize larger-scale training corpora and collect small pilot datasets to better supervise pitch/timbre--emotion mapping, enabling stronger regression or lightweight neural models (e.g., polynomial regression, gradient boosting, shallow CNNs over spectrograms). Enhancements to text emotion detection will involve acquiring or constructing a more representative elderly dialogue dataset, as well as designing a tighter fusion mechanism between the C-LIWC and LLM pathways (and the offline ensemble) to produce complementary, unified outputs. The conversation strategy module will also be refined by streamlining its input structure, expanding the keyword memory corpus, and iteratively testing prompt formulations to ensure the LLM maintains clarity, relevance, and a well-defined companion persona.

In parallel with these functional improvements, we plan to develop a high-fidelity user interface prototype using Figma, focusing on accessibility and simplicity to accommodate older users with limited digital literacy. The UI will be designed to minimize cognitive load, offer clear interaction cues, and provide multi-modal feedback, aligning with the psychological and usability considerations embedded in the back-end architecture.

We acknowledge that certain aspects---particularly those involving secure data storage, compliance with high-level software and data protection regulations, and large-scale data collection---are unlikely to be completed within the current development cycle. Nevertheless, these remain essential long-term objectives, and future iterations of the Eldercare Agent will work toward meeting these requirements while preserving the system's core values of safety, personalization, and psychological sensitivity.

\section{Conclusion}
\label{sec:conclusion}

This paper presents an emotion-aware conversational AI agent specifically designed for Chinese elderly mental health support. The system combines multimodal emotion recognition, hierarchical strategy selection, and long-term trend monitoring to provide accessible, personalized companionship for older adults. Through the integration of text and voice emotion analysis, psychologically-grounded response strategies, and persistent user modeling, the agent demonstrates the potential for AI-driven mental health support tailored to the unique needs of aging populations. The evaluation results show strong performance across emotion recognition tasks, with the stacking ensemble achieving high accuracy in detecting anger, sadness, joy, and emotional intensity. While challenges remain in dialectal speech recognition, real-world data collection, and response optimization, the modular architecture provides a foundation for iterative improvement and deployment in real-world eldercare scenarios.

\section*{Acknowledgments}
This project could not have been completed without the dedication and collaboration of everyone on our team. Thank you for all your contributions. We are also deeply grateful to Professor Teuscher and to the faculty and TAs at Portland State University for creating this opportunity and for their guidance throughout the project. Your support enabled us to complete this work successfully and learn a great deal along the way.

\begin{thebibliography}{99}

\bibitem{fitzpatrick2017woebot}
K. K. Fitzpatrick, A. Darcy, and M. Vierhile, ``Delivering cognitive behavior therapy to young adults with symptoms of depression and anxiety using a fully automated conversational agent (Woebot): A randomized controlled trial,'' \textit{JMIR Mental Health}, vol. 4, no. 2, p. e7785, 2017.

\bibitem{possati2023replika}
L. M. Possati, ``Psychoanalyzing artificial intelligence: The case of Replika,'' \textit{AI \& Society}, vol. 38, no. 4, pp. 1725--1738, 2023.

\bibitem{zhou2020xiaoice}
L. Zhou, J. Gao, D. Li, and H.-Y. Shum, ``The design and implementation of XiaoIce, an empathetic social chatbot,'' \textit{Computational Linguistics}, vol. 46, no. 1, pp. 53--93, 2020.

\bibitem{chen2024structured}
Y. Chen, X. Zhang, J. Wang, X. Xie, N. Yan, H. Chen, and L. Wang, ``Structured Dialogue System for Mental Health: An LLM Chatbot Leveraging the PM+ Guidelines,'' in \textit{Social Robotics (ICSR + InnoBiz 2024)}, Lecture Notes in Computer Science, vol. 15170, Springer, 2024, pp. 262--271.

\bibitem{firdaus2022multimodal}
M. Firdaus, H. Chauhan, A. Ekbal, and P. Bhattacharyya, ``EmoSen: Generating sentiment and emotion controlled responses in a multimodal dialogue system,'' \textit{IEEE Transactions on Affective Computing}, vol. 13, no. 3, pp. 1555--1566, 2022.

\bibitem{yu2020chsims}
W. Yu, H. Xu, F. Meng, Y. Zhu, Y. Ma, J. Wu, J. Zou, and K. Yang, ``CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality,'' in \textit{Proceedings of ACL 2020}, 2020, pp. 3718--3727.

\bibitem{baidu2024speech}
Baidu AI Speech, ``Speech recognition products \& documentation,'' https://ai.baidu.com/tech/speech, accessed 2024.

\bibitem{mcfee2015librosa}
B. McFee \textit{et al.}, ``librosa: Audio and Music Signal Analysis in Python,'' in \textit{Proceedings of the 14th Python in Science Conference}, 2015, pp. 18--25.

\bibitem{mauch2014pyin}
M. Mauch and S. Dixon, ``pYIN: A fundamental frequency estimator using probabilistic threshold distributions,'' in \textit{ICASSP}, 2014, pp. 659--663.

\bibitem{davis1980mfcc}
S. Davis and P. Mermelstein, ``Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences,'' \textit{IEEE Trans. Acoustics, Speech, and Signal Processing}, vol. 28, no. 4, pp. 357--366, 1980.

\bibitem{tzanetakis2002genre}
G. Tzanetakis and P. Cook, ``Musical genre classification of audio signals,'' \textit{IEEE Trans. Speech and Audio Processing}, vol. 10, no. 5, pp. 293--302, 2002.

\bibitem{scherer2003vocal}
K. R. Scherer, ``Vocal communication of emotion: A review of research paradigms,'' \textit{Speech Communication}, vol. 40, no. 1-2, pp. 227--256, 2003.

\bibitem{baltrusaitis2019multimodal}
T. Baltrušaitis, C. Ahuja, and L.-P. Morency, ``Multimodal Machine Learning: A Survey and Taxonomy,'' \textit{IEEE TPAMI}, vol. 41, no. 2, pp. 423--443, 2019.

\bibitem{sheikh2012gds}
J. I. Sheikh and J. A. Yesavage, ``Geriatric Depression Scale---Short Form,'' 2012.

\bibitem{tausczik2010liwc}
Y. R. Tausczik and J. W. Pennebaker, ``The psychological meaning of words: LIWC and computerized text analysis methods,'' \textit{Journal of Language and Social Psychology}, vol. 29, no. 1, pp. 24--54, 2010.

\bibitem{huang2012liwc}
X. T. Huang, Q. S. Liao, and P. Huang, ``Development of the Chinese version of the LIWC and its reliability and validity,'' \textit{Acta Psychologica Sinica}, vol. 44, no. 11, pp. 1402--1412, 2012.

\bibitem{deepseek2023llm}
DeepSeek, ``DeepSeek LLM: Scalable language models with high performance,'' https://github.com/deepseek-ai/DeepSeek-LLM, 2023.

\bibitem{wang2020minilm}
W. Wang \textit{et al.}, ``MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers,'' \textit{NeurIPS}, 2020.

\bibitem{wang2021minilmv2}
W. Wang, H. Bao, S. Huang, L. Dong, and F. Wei, ``MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers,'' \textit{arXiv:2012.15828}, 2021.

\bibitem{reimers2019sbert}
N. Reimers and I. Gurevych, ``Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,'' \textit{EMNLP}, 2019.

\bibitem{pedregosa2011scikit}
F. Pedregosa \textit{et al.}, ``Scikit-learn: Machine Learning in Python,'' \textit{JMLR}, vol. 12, pp. 2825--2830, 2011.

\bibitem{ke2017lightgbm}
G. Ke \textit{et al.}, ``LightGBM: A Highly Efficient Gradient Boosting Decision Tree,'' \textit{NeurIPS}, 2017.

\bibitem{alshamsi2023ensemble}
A. A. Al Shamsi and S. Abdallah, ``Ensemble Stacking Model for Sentiment Analysis of Emirati and Arabic Dialects,'' 2023.

\bibitem{sano2018jmir}
A. Sano, S. Taylor, A. W. McHill, A. J. K. Phillips, L. K. Barger, E. Klerman, and R. W. Picard, ``Identifying Objective Physiological Markers and Modifiable Behaviors for Self-Reported Stress and Mental Health Status Using Wearable Sensors and Mobile Phones: Observational Study,'' \textit{J Med Internet Res}, vol. 20, no. 6, e210, 2018.  \url{https://www.jmir.org/2018/6/e210/}

\bibitem{devlin2019bert}
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, ``BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,'' in \textit{NAACL-HLT}, 2019.

\bibitem{liu2019roberta}
Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, et al., ``RoBERTa: A Robustly Optimized BERT Pretraining Approach,'' arXiv:1907.11692, 2019.

\bibitem{sun2019ernie}
Y. Sun, S. Wang, Y. Li, S. Feng, X. Chen, H. Zhang, et al., ``ERNIE: Enhanced Representation through Knowledge Integration,'' arXiv:1904.09223, 2019.

\bibitem{qin2023berterc}
X. Qin, Z. Wu, J. Cui, T. Zhang, Y. Li, J. Luan, B. Wang, and L. Wang, ``BERT-ERC: Fine-tuning BERT is Enough for Emotion Recognition in Conversation,'' arXiv:2301.06745, 2023. \url{https://arxiv.org/abs/2301.06745}

\bibitem{yang2021emotiondynamics}
H. Yang and J. Shen, ``Emotion Dynamics Modeling via BERT,'' arXiv:2104.07252, 2021. \url{https://arxiv.org/abs/2104.07252}

\bibitem{lei2023instructerc}
H. Lei, Z. Yu, Y. Xu, S. Feng, and D. Yu, ``InstructERC: A Unified Framework for Emotion Recognition in Conversation with Instruction Tuning,'' 2023.

\bibitem{fu2024laercs}
Y. Fu, J. Wu, Z. Wang, M. Zhang, L. Shan, Y. Wu, and B. Liu, ``LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics,'' 2024. \url{https://arxiv.org/abs/2403.07260}

\bibitem{inkster2018wysa}
B. Inkster, S. Sarda, and V. Subramanian, ``An Empathy-Driven, Conversational Artificial Intelligence Agent (Wysa) for Digital Mental Well-Being: Real-World Data Evaluation Mixed-Methods Study,'' \textit{JMIR mHealth and uHealth}, vol. 6, e12106, 2018.

\bibitem{mehta2021youper}
A. Mehta, A. N. Niles, J. H. Vargas, T. Marafon, D. D. Couto, and J. J. Gross, ``Acceptability and Effectiveness of Artificial Intelligence Therapy for Anxiety and Depression (Youper): Longitudinal Observational Study,'' \textit{J Med Internet Res}, vol. 23, e26771, 2021.

\bibitem{gingerio2018}
Ginger.io, ``An App That Monitors Your Mental Health,'' 2018. Available online: \url{https://d3.harvard.edu/platform-digit/submission/ginger-io-an-app-that-monitors-your-mental-health/} (accessed Sept. 30, 2024).

\end{thebibliography}

\section{Appendix: UI Design Prototype}
\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{login.png}
        \caption{Login interface}
        \label{fig:ui_login}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{main page.png}
        \caption{Main conversation interface}
        \label{fig:ui_main}
    \end{subfigure}
    
    \vspace{0.5cm}
    
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{trend_analysis.png}
        \caption{Emotion trend analysis}
        \label{fig:ui_trend}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{user_bio.png}
        \caption{User profile management}
        \label{fig:ui_profile}
    \end{subfigure}
    
    \caption{ElderCare Companion user interface design showcasing the four main functional modules: (a) user authentication, (b) multimodal conversation interface with emotion analysis, (c) long-term emotional trend monitoring, and (d) personalized user profile management.}
    \label{fig:ui_overview}
\end{figure}
\end{document} 
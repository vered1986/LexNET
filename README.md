# LexNET: Integrated Path-based and Distributional Method for Lexical Semantic Relation Classification

**LexNET** is an open source framework for classifying semantic relations between term-pairs. It uses distributional information on each term, and path-based information, encoded using an LSTM.

If you use **LexNET** for any published research, please include the following citation:

<b>"Path-based vs. Distributional Information in Recognizing Lexical Semantic Relations"</b><br/>
Vered Shwartz and Ido Dagan. Proceedings of the 5th Workshop on Cognitive Aspects of the Lexicon (CogALex-V), in COLING 2016. [link](http://arxiv.org/abs/1608.05014)

**LexNET** participated in the CogALex-V Shared Task on the Corpus-Based Identification of Semantic Relations, and achieved the best performance among the competitors on subtask 2 (semantic relation classification). The following paper describes the submission:

<b>"CogALex-V Shared Task: LexNET - Integrated Path-based and Distributional Method for the Identification of Semantic Relations"</b><br/>
Vered Shwartz and Ido Dagan. Proceedings of the 5th Workshop on Cognitive Aspects of the Lexicon (CogALex-V), in COLING 2016. [link](https://arxiv.org/abs/1610.08694)

***

To start using **LexNET**, read the [Quick Start](https://github.com/vered1986/LexNET/wiki/Quick-Start) or the [Detailed Guide](https://github.com/vered1986/LexNET/wiki/Detailed-Guide).

***

## Version 2:

### Major features and improvements:
* Using dynet instead of pycnn
* Making the resource creation time and memory efficient

### Bug fixes:
* Too many paths in parse_wikipedia (see issue [#2](https://github.com/vered1986/HypeNET/issues/2))

### Notes:
* To reproduce the results reported in the paper, please use [V1](https://github.com/vered1986/LexNET/tree/v1).
* The pre-processed [corpus files](https://drive.google.com/file/d/0B0kBcFEBhcbha2N0Vm1FYW01Umc/view?usp=sharing) are available for V2 now as well!
***

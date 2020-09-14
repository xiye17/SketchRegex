# SketchRegex
Data and code for the paper ["Sketch-Driven Regular Expression Generation from Natural Language and Examples
"](https://arxiv.org/abs/1908.05848).

 ```
 @InProceedings{sketchregex,
  title = {Sketch-Driven Regular Expression Generation from Natural Language and Examples},
  author = {Xi Ye, Qiaochu Chen, Xinyu Wang, Isil Dillig, and Greg Durrett},
  booktitle = {Transactions of the Association for Computational Linguistics (TACL)},
  year = {2020},
}
 
 ```
 
 ## Prerequisites
 
 * pytorch > 1.0.0
 * [Z3](https://github.com/Z3Prover/z3). Make sure you have Z3 installed with the Java binding.
 * JAVA 1.8.0
 
 ## Code
 
 Our sketch-driven framework can be instatiated with either a neural parser [`DeepSketch`],or a grammar-based parser [`GrammarSketch`] (coming soon). Please refer to the **README** of each module for details.

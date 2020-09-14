# DeepSketch -- SketchRegex

Code for DeepSketch and re-implementation of [DeepRegex](https://arxiv.org/abs/1608.03000) and [SemRegex] (https://www.aclweb.org/anthology/D18-1189/)(refered as ***DeepRegex+MML*** in our paper) as baselines
 
 ## Run Baselines
 
 **Using Pretrained Models**
 
 **1.** generate the k-best list of regexes for each description from a given dataset.
 
 `python decode.py <dataset> <model_id> --split test`.
 
 * `<dataset>`: the target dataset, can be **Turk** or **KB13**.
 * `<model_id>`: the id of pretrained model, corresponds to the checkpoint at `checkpoints/<dataset>/<model_id>.tar`.
 
 E.g., the command [`python decode.py Turk pretrained-MLE --split test`] will produce decode files at `decodes/Turk/test-pretrained-MLE`. Each decode file is a readable text file.
 
 **2.** evaluate semantic accuracy
 
  `python eval.py <dataset> <model_id> --split test`
  
  E.g., the command [`python eval.py Turk pretrained-MLE --split test`] will evaluate decode files at `decodes/Turk/test-pretrained-MLE` using semantic accuracy, which is based on DFA-equivelance.
  The optiional 'do_filter` flag enables evaluation with filtering mechanism (See ***DeepRegex+Filter*** in the paper).
  
   **Retrain Models**
  
  To train your own model with ***MLE*** objective, run
  
   `python train.py <dataset> --model_id <model_id>`.
   
   The models will be stored in `checkpoints/<dataset>` directory with names following `<model_id>*.tar`.
   To enable ***MML*** training, use the flag `--do_rl`. Refer to the code for details of more optional arguments. 


 ##  Run Sketch-Driven Approaches
 
 The sketch version datasets are **TurkSketch** and **KB13Sketch**.
 
 **1.** generate the k-best list of sketches.
 
 `python decode.py <dataset> <model_id> --split test`.
 
 * `<dataset>`: the target dataset, can be **TurkSketch** or **KB13Sketch**.
 
 E.g., the command [`python decode.py TurkSketch pretrained-MLE --split test`].
  
 **2.** evaluate semantic accuracy using synthesizer
 
  `python eval.py <dataset> <model_id> --split test`
  
   E.g., the command [`python eval.py TurkSketch pretrained-MLE --split test`].

The evaluation script will recoginize the dataset is using sketch (by the dataset name), and automatically call the synthesizer (a JAR at `external/resnax.jar`) to synthesize the sketches.

## Cache
We note that evaluating the DFA-equivelenace and calling synthesizer with python `subprocess` can be time-consuming, so we create caches to avoid repeatedly evaluating the same regex pair or the same sketch. Those caches will be stored in `caches/` .
  
 
 

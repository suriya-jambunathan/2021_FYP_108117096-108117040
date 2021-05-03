# Final Year Project 2021 - Suriya Prakash Jambunathan, Harish Raj D.R.
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

This is the Final Year Project 2021 of Final Year Students of Electronics and Communication Engineering Department, National Institute of Technology, Tiruchirapalli

Suriya Prakash Jambunathan, 108117096
Harish Raj D.R., 108117040


[embed]http://example.com/file.pdf[/embed]




MACHINE LEARNING APPLICATION IN
ANTENNA
A Thesis submitted in partial fulfillment of the requirementsfor the
award of the degree of
B.Tech
in
Electronics and Communication Engineering
By
Harish Raj.D.R. (108117040)
Suriya Prakash Jambunathan(108117096)
ELECTRONICS AND COMMUNICATION ENGINEERING
NATIONAL INSTITUTE OF TECHNOLOGY
TIRUCHIRAPALLI- 620015
APRIL 2021
BONAFIDE CERTIFICATE
This is to certify that the project titled MACHINELEARNING APPLICATION IN
ANTENNA is a bonafide record of the work done by

Harish Raj.D.R. (108117040)
Suriya Prakash Jambunathan(108117096)
in partial fulfillment of the requirements for theaward of the degree of Bachelor of
Technology of the NATIONAL INSTITUTE OF TECHNOLOGY,
TIRUCHIRAPPALLI, during the year 2017-2021.

Dr. D. Sriram Kumar Dr. P. Muthuchidambaranathan
Guide Head of the Department
Viva-voce held on 13.04.
Internal Examiner External Examiner
ABSTRACT
Micro-strip patch antennas arepredominantly used inmobile communicationand
healthcare.Theirperformancesareevenimproved,usingSplit-RingResonatorcells.
But, forfindingtheidealdimensionsofthemicro strippatchantenna, aswellas
findingtherightnumberandsizeofthesplitringresonatorcells,consumesalotof
time, whenweuseElectromagneticSimulationsoftwaresto design first,andthen
simulate.Usingthepre-calculatedresultsofcertainsetofmicrostrippatchantennas
withsplitringresonators,amachinelearningmodelcanbetrained,andhencebeused
topredicttheantennametrics,whenthedimensionsarespecified.Whenthemachine
learning algorithms are combined with feature-optimisation algorithms such as
Genetic Algorithm, the efficiency and performancecan be improved, further.

Keywords : Micro-stripPatch Antenna; Split Ring Resonator, Genetic algorithm,
Machine Learning

ACKNOWLEDGEMENTS
Wewouldliketoexpressourdeepestgratitudetothefollowingpeopleforguidingus
throughthiscourseandwithoutwhomthisprojectandtheresultsachievedfromit
would not have reached completion.

Dr.D.SriramKumar , Professor, DepartmentofElectronicsand Communication
Engineering,forhelpingusandguidingusinthecourseofthisproject.Withouthis
guidance, wewouldnot havebeenable tosuccessfullycompletethisproject.His
patience and genial attitude is and always will bea source of inspiration to us.

Dr. P. Muthuchidambaranathan , the Head of the Department, Department of
ElectronicsandCommunicationEngineering,forallowingustoavailthefacilitiesat
the department.

CHAPTER 1
INTRODUCTION
1.1 Objective and Goal
The objective of this project is to implement several machine learning
algorithmstopredictthebandwidth,GainandVoltageStandingWaveRatio(VSWR)
ofanSplitRingResonator(SRR),basedonanexistingdatasheetandtoidentifythe
bestmachinelearningalgorithmtofindtheabovementionedparametersinafaster,
efficient way with maximum accuracy and to study which features arebest in
predictionoftheBandwidth,gainandVSWRofaSRRantennainordertooptimize
the prediction.

Themaingoaloftheprojectistodevelopamachinelearningmodelwhich
canpredictBandwidth,gainandVSWRinafaster,efficientandmostaccurateway
possiblewhencomparedtoconventionalmethodssuchasHFSSandCSTsimulations
which are slow and resource heavy.

1.2 Thesis Motivation
Today,tofindthebandwidth,VSWRandgaintherearemanyways.Theyave
their own advantages and disadvantages. One method is by using mathematical
calculationsusingtheparameters.Whilethismethodcanbesimpleforcertaintypes
ofantennae,itcanquicklybecometediousanddifficulttocalculateandverytime
consuming.Thoughcomputerlanguagessuchaspythoncanbeusedforthesetedious
calculations,quickchangesinthevaluescanbedifficultsincetherearehundredsof
linesofcodeandtheyarepronetohumanerrors.SmithChartscanhelptofindthe
VSWR, Gain and bandwidth in an indirectway by findingtheinputand output
impedancesandbyusingformulae.Eventhismethodifnotdonecarefullycanleadto
errors.

Various3DelectromagneticanalysissoftwaressuchasCSTandHFSScan
helptofindtheseparameters.Whilethesemethodsproduceaccuratevalues,theyare
oftentimeconsumingandrequirelearningthesoftwarewhichcaniselfconsumealot
of time due tovarious terminologiesand optionseachsoftware offers.Theyalso

requiretobuildtheantennafromgroundupinan3Denvironmentandthensimulate
whichisresourceheavyanddependingontheantennatypeandit’sdimensions,itcan
take a few minutes to days to finish.

This projectwillhopefullyeliminatethesetediousprocessesandprovidea
straightforward method for calculations of bandwidth,VSWR and gain of antenna.

CHAPTER 2
LITERATURE REVIEW
CHAPTER 3
SPLIT-RING RESONATOR
Class_Reg Algorithm

A simple classification algorithm applied to predictthe Bandwidth, Gain or
VSWR values wouldn’t be able to predict the valueto a greater accuracy, because a
classification algorithm doesn’t work with continuousdata, so we will be able to
predict if a value is in a particular range, but notthe exact value. For such cases,
where the prediction of a particular floating pointvalue is required, a regression
algorithm is used. When the regression algorithm wasimplemented on predicting the
value, due to the data being unbalanced, it resultedin a not-so-good accuracy in
prediction.

So, we formulated an algorithm, wherein, we use bothclassification and
regression in predicting the values.

On implementing a machine learning algorithm, thedataset consists of two
halves, the independent features, and dependent variable.The independent features
are the ones using which we will train our machinelearning algorithm, and predict.
The dependent variable is the output, which we willbe using to train the machine
learning algorithm, and use for validating the accuracywhile testing the algorithm.

In the Class_Reg algorithm, after splitting into atraining set and testing set,
the dependent variables in the training set are segregatedinto a user-defined number
of classes. In the same corresponding way, the independentfeatures are also
segregated, based on the corresponding dependent variablevalue. A Regression
algorithm is trained on each of these portions ofthe training set. Concurrently, a
Classification algorithm is also trained, to predictwhich particular portion, does the
split part belong to in the entire dataset.

For Example, let the dataset be split into trainingset and testing set. Then, the
training set is split into, say, Class A, B, C, Dand E. Now, we have five different
portions of the training set. On each of these differentportions, we train a regression
algorithm. On the whole training set, we train a classificationalgorithm, to predict if
the particular set of independent features, correspondsto Class A, B, C, D or E. On
applying the Class_Reg algorithm on the testing set,we supply the independent
features to the algorithm, on which we apply the trainedclassification algorithm to
predict which class it belongs to. Using the name/numberof the class, we will use the
particular class’s regression algorithm, to predictthe exact value of the parameters,
Bandwidth, Gain, and VSWR.

As the method of class_reg algorithm is clear [Figure ], there is no clear theory
to know which kind of classification or regression algorithm would be suitable in the
prediction of our parameters. So, we use many numberof classifiers and regressors,
and use all different classifier-regressor combinationsto train on the dataset, and test
on the dataset, and use the testing accuracy, as ametric to find out which of the
classifier-regressor combination, will work best forour dataset.

The metric we have used to evaluate the performanceis weighted mean
absolute percentage error [ref], which gives us theaverage of all the percentage error
of each prediction with its original value.

Add formula here
The above stated method can find us the best classifier-regressorcombination,
but there is a possibility that the particular combinationonly works best for the
particular train-test split. So, in order to findthe actual performance metric for all the
classifier-regressor combinations, we consider fivedifferent training testing splits, and
average the performance metric of all of them fora particular classifier-regression
combination. Then, the best classifier-regressor isfound out by checking which of
them has the best metric.

We split the dataset into a training set, validationset and testing set. The
testing set is not involved, until all the procedureof finding the best
classifier-regressor combination is found out usingthe different combinations of the
training set and the validation set.

This method, even though may seem to take more timethan the conventional
classification or regression algorithms, it has helpedin achieving the best of both
classification and regression algorithms, and we’vegot better accuracy(100 - error %)
using class_reg algorithm compared to a conventionalregression algorithm.

Genetic Algorithm

A Genetic Algorithm [ref] is a feature optimizationalgorithm based on the
theory of Evolution. According to the Theory of Evolution,best features in an
organism are passed on to the subsequent generations,and eventually in future, after
several generations, those best features would bedominant in the organism [ref].

Genetic algorithm is one of the oldest but a veryefficient feature optimization
algorithm,it uses probabilistic transition rules,and not deterministic rules,
which gives it an edge over some of the other featureoptimization algorithms.

Add image of genetic algorithm flowchart here
In Genetic Algorithm [fig ], initially a random population of chromosomes
( the genetic blueprint which is unique to every individual ) based on the defined
population size. The chromosomes in this case, wouldbe an array of True and False ,
denoting whether the particular independent featureis taken into consideration in the
current iteration of the algorithm, or not. The individuals(chromosomes) in the
population are evaluated by training an user-definedalgorithm on the training set, and
testing it on the validation set, and the particularchromosome is scored based on a
fitness function, which can be defined by the user.The chromosomes are then ordered
based on the fitness score, with, higher the fitnessscore, meaning higher chance of
survival, hence a prefered trait.

The sorted population is then returned to the originalfunction, where the
parents for creating the next generation are chosenbased on the user-defined number
of parents. The sorted list of parents are then bredin a roulette basis, that is, a child
gets half of the chromosomes from one parent, andthe other half from the other
parent. After we get the set of all the children,they are mutated according to a
user-defined mutation rate, where the True or Falsein the chromosome, are mutated
(altered) to False or True, randomly, based on themutation rate. This step is done in
order to ensure the reliance with nature, where slightmutation occurs randomly,
which is one of the keys to the theory of evolution.

The set of mutated children constitute the next generation.The next generation
population is then evaluated and sorted accordingto the user-defined fitness function,
and the required number of parents are chosen, andbred to give children, which are
then mutated to form the next generation.

The steps defined above are continued, until the user-definednumber of
generations has been reached, which is the terminationcondition for the algorithm.

Genetic Algorithm applied in conjunction with Class_reg Algorithm

For the calculation of the fitness score in GeneticAlgorithm, there is a need to
supply a function, which will be evaluated based onthe fitness score. So, we have
supplied the Class_reg function as the function onwhich chromosome selected list of
independent features are trained on the training set,and evaluated using the fitness
score, on the validation set. For the fitness functionin the genetic algorithm, we have
used five different metrics, namely, weighted meanabsolute percentage error [ref],
mean squared error [ref], r2 score [ref], weightedmean absolute percentage error
multiplied with mean squared error, and weighted meanabsolute percentage error
added with mean squared error.

Add image of our project flowchart here
After the Genetic Algorithm is run, we can run theClass_reg algorithm again
on the selected list of independent features, andthe performance can be evaluated on
the testing set.

CHAPTER
RESULTS AND DISCUSSION
On applying the Class_reg algorithm for predictingthe dependent parameters,
Bandwidth, Gain and VSWR, we have presented the bestfive classifiers and
regressors for each of the parameters in the tablesbelow.

Add table of bandwidth table from review 1 here
According to the results [fig] on applying the Class_regalgorithm for
predicting bandwidth, the best classifier is DecisionTree Classifier, and the best
regressor is Gradient Boosting Regressor, with a weightedmean absolute percentage
error of 0..

Decision Tree Classifier
It is a non-parametric supervised learning method,whose goal is to
create a model that predicts the value of a targetvariable by learning simple decision
rules

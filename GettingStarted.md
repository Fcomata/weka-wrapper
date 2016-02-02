# Introduction #

Weka, if you don't know already, is a hugely useful toolkit for machine learning and data mining applications.

My one gripe with Weka is that it can be a pain to program with. It is fine if you are using it via the GUI interface, but calling it from within Java code can be a nightmare.

The Weka Wrapper project aims to make it much easier to write readable, robust code around Weka. Have a look at the example below to see how much of a difference it can make.

# Look at the pretty flowers #

For starters, lets have a look at using the Weka Wrapper on an old favourite: the [iris dataset](http://dtai.cs.kuleuven.be/DataMiningInPractice11/DATASET/iris.arff).

This dataset contains the measurements (_sepal length_, _sepal width_, _petal length_ and _petal width_) and type (_Iris Setosa_, _Iris Versicolour_ or _Iris Virginica_) of a few hundred different plants. The task is to learn how to identify the type of plant automatically from their measurements.

## Initial Setup ##

First, set up an enum to keep track of the features the classifier will work with

```
	enum IrisFeatures {sepallength,sepalwidth,petallength,petalwidth} ;
```

Because this is a multiclass problem, we also need another enum to keep track of the classes our classifier will choose from. This would be unnecessary if the class attribute was numeric or binary.

```
	enum IrisClass {IrisSetosa,IrisVersicolor,IrisVirginica} ;
```

Next, create a Decider which will be responsible for classifying the flowers. There is a nice builder interface for doing this, which is handy because there can sometimes be a lot to configure.

```

	Decider<IrisFeatures,IrisClass> decider = 
		new DeciderBuilder("IrisDecider", IrisFeatures.class)
			.setDefaultAttributeTypeNumeric()
			.setClassAttributeTypeEnum("class", IrisClass.class)
			.build();
```

That bit of code produces a decider that knows
  * what attributes are involved, so it will not accept attributes that it doesn't know about)
  * what data types these attributes are (in this case all numeric, but boolean, string and enum attributes are also possible), so it will complain if you suddenly try to use something else.
  * what the expected output is (in this case, a member of the IrisClass enum, but numeric or boolean classes are also possible)

## Training the decider ##

At this point the decider knows what decisions you want it to make, but it doesn't know how to make them. You need to feed it some training examples.

First, create a dataset.
```
	Dataset<IrisFeatures,IrisClass> dataset = decider.createNewDataset() ;
```

Again, this dataset knows what types of features and class values are expected, and so is in a position to complain if you try to do something stupid.

Unfortunately it is also empty. We could build it programatically (more on this below), but lets just load it from file:

```
	dataset.load(new File("path/to/iris.arff")) ;  
```

And use it to train up an SMO classifier:

```
	decider.train(new SMO(), dataset) ;
```


## Classifying flowers ##

Now that the decider is all trained up, it's ready to look at the pretty flowers!

First we need to make a flower. We can ask for an instance builder from the decider to make that easy.

```
	Instance flower = decider.getInstanceBuilder()
		.setAttributeMissingResponse(BuildResponse.THROW_ERROR)
		.setAttribute(IrisFeatures.sepallength, 6.4)
		.setAttribute(IrisFeatures.sepalwidth, 3.1)
		.setAttribute(IrisFeatures.petallength, 5.5)
		.setAttribute(IrisFeatures.petalwidth, 1.8)
		.build() ;
```

The nice thing about the code above is that it all gets verified. It will throw compiler errors if you try to set an unknown attribute, and will throw runtime errors if any attributes are unspecified (this is configurable), or if you try to set an attribute value as numeric when it is supposed to be boolean, and so on.

So, what is the flower's type?

```
	IrisClass flowerType = decider.getDecision(flower) ;
```

What is the probability distribution for the decision?

```
	HashMap<IrisClass, Double> distributions = 
		decider.getDecisionDistribution(i) ;
```

## Building training data programatically ##

We can also get an instance builder from the dataset, so it is easy to create training and testing sets within the code rather than loading them from file.

```
	Instance flower2 = dataset.getInstanceBuilder()
		.setAttributeMissingResponse(BuildResponse.THROW_ERROR)
		.setClassAttributeMissingResponse(BuildResponse.THROW_ERROR)
		.setAttribute(IrisFeatures.sepallength, 4.3)
		.setAttribute(IrisFeatures.sepalwidth, 1.9)
		.setAttribute(IrisFeatures.petallength, 6.2)
		.setAttribute(IrisFeatures.petalwidth, 3.1)
		.setClassAttribute(IrisClass.IrisSetosa)
		.build() ;
            
	dataset.add(flower2) ;
```


## Saving and loading models ##

Once you have a decider that works well, it is a good idea to save it.

```
	decider.save(new File("path/to/decision.model")) ;
```

That way, you won't need a training dataset next time. You can just do this immediately after creating the decider.

```
	decider.load(new File("path/to/decision.model")) ;
```

## Saving datasets ##

If you build up a dataset, it is a very good idea to save it as an arff file. That way you will be able to load it up in the Weka GUI, and do lots of experiments quickly to come up with a classifier that works well.

```
	dataset.save(new File("path/to/dataset.arff")) ; 
```
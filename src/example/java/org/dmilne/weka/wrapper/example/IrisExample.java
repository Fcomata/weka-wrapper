package org.dmilne.weka.wrapper.example;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.dmilne.weka.wrapper.Decider;

import weka.classifiers.functions.SMO;
import weka.core.Instance;
import org.dmilne.weka.wrapper.*;
import org.dmilne.weka.wrapper.InstanceBuilder.BuildResponse;

public class IrisExample {

	private enum IrisFeatures {sepallength,sepalwidth,petallength,petalwidth} ;
	private enum IrisClass {IrisSetosa,IrisVersicolor,IrisVirginica} ;

	public static void main(String[] args) throws Exception {

		//build a decider, which knows:
		// - what attributes are involved
		// - what data types these attributes are (in this case all numeric)
		// - what the expected output is (in this case, an enum, but numeric and binary is also doable)
		Decider<IrisFeatures,IrisClass> decider = 
			new DeciderBuilder("IrisDecider", IrisFeatures.class)
		.setDefaultAttributeTypeNumeric()
		.setClassAttributeTypeEnum("class", IrisClass.class)
		.build();

		//load training data from file
		//this will check that attributes match IrisFeatures enum, that class attribute is named "class" and is of correct type, and so on.
		Dataset<IrisFeatures,IrisClass> dataset = decider.createNewDataset() ;
		dataset.load(new File("src/example/resources/iris.arff")) ;

		//train a classifier using loaded training data.
		decider.train(new SMO(), dataset) ;

		//save the classifier so we could skip training in future
		//unfortunately this doesn't make any checks to see if classifier was trained on expected attributes
		//any idea how one would do that?
		decider.save(new File("src/example/resources/iris.model")) ;

		//load the classifier saved previously
		//unfortunately this doesn't make any checks to see if classifier was trained on expected attributes
		decider.load(new File("src/example/resources/iris.model")) ;

		//build an instance that we can classify
		//this will check that all attributes are set (optional) and that values are the correct type.
		Instance i = decider.getInstanceBuilder()
		.setAttributeMissingResponse(BuildResponse.THROW_ERROR)
		.setAttribute(IrisFeatures.sepallength, 6.4)
		.setAttribute(IrisFeatures.sepalwidth, 3.1)
		.setAttribute(IrisFeatures.petallength, 5.5)
		.setAttribute(IrisFeatures.petalwidth, 1.8)
		.build() ;

		//note: building training instances algorithmically is done in much the same way

		//now identify the class of the instance we built
		IrisClass c = decider.getDecision(i) ;
		System.out.println(c) ;

		//and get some details about how this decision was made 
		HashMap<IrisClass, Double> distributions = decider.getDecisionDistribution(i) ;
		for (Map.Entry<IrisClass, Double> e:distributions.entrySet()) 
			System.out.println(e.getKey() + ": " + e.getValue()) ;            
	}
}

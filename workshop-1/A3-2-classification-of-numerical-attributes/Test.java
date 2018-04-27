import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

public class Test {

    public static void main(String[] args) {

        try {

            DataSource source = new DataSource("../../datasets/Iris/iris.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            NaiveBayes classifier = new NaiveBayes();
            classifier.buildClassifier(data);

            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));
            System.out.println(evaluation.toSummaryString());

        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
import java.util.Random;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Test {

    public static void main(String[] args) {

        try {

            DataSource source = new DataSource("../../datasets/Wikipedia_70/wikipedia_70.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            String[] options = new String[2];
            options[0] = "-R";
            options[1] = "1";
            StringToWordVector filter = new StringToWordVector();
            filter.setOptions(options);
            filter.setInputFormat(data);

            Instances newData = Filter.useFilter(data, filter);

            NaiveBayesMultinomial classifier = new NaiveBayesMultinomial();
            classifier.buildClassifier(newData);

            Evaluation evaluation = new Evaluation(newData);
            evaluation.crossValidateModel(classifier, newData, 10, new Random(1));
            System.out.println(evaluation.toSummaryString());

        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
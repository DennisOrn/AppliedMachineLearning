import weka.classifiers.lazy.IBk;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.core.Instances;

public class Test {

    public static void main(String[] args) {

        try {

            DataSource source = new DataSource("../../datasets/GPUbenchmark/GPUbenchmark.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            for (int k = 1; k <= 10; ++k) {

                double differenceSum = 0;

                for (int i = 0; i < data.numInstances(); ++i) {

                    Instances newData = source.getDataSet();
                    newData.setClassIndex(newData.numAttributes() - 1);

                    // Slightly more accurate by removing the attribute on index 1.
                    newData.deleteAttributeAt(1);

                    Instance removedInstance = newData.instance(i);
                    double removedInstanceBenchmark = removedInstance.value(removedInstance.numValues() - 1);
                    newData.delete(i);

                    IBk classifier = new IBk(k);
                    classifier.buildClassifier(newData);

                    double predictedBenchmark = classifier.classifyInstance(removedInstance);
                    double difference = Math.abs(removedInstanceBenchmark - predictedBenchmark);
                    differenceSum += difference;
                }

                double averageDifference = differenceSum / data.numInstances();

                System.out.println("k = " + k + ", average difference: " + averageDifference);
            }

        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
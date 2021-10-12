using System;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
namespace LogisticRegressionUsingAccord
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======LOGISTIC REGRESSION USING ACCORD ============");
            string fileName = "../../../winequality-red.txt";
            //Ucitavanje podataka
            Console.WriteLine("Loading Data...");
            double[][] data = Helper.MatrixLoad(fileName, true, ';');
			//Fisher-Yates algoritam za permutaciju observacija u skupu podataka. Menjamo raspored observacija u skupu
            //Helper.randomize(data);
            double[] allLabels = Helper.ExtractLabels(data);
            data = Helper.RemoveLabels(data);
            int[] allLabelsI = Helper.ConvertDoubleToInt(allLabels);
            Console.WriteLine("Feature selection");
            data=FeatureSelection.selectFeatures(data, allLabelsI,3);

			//70% podataka koristimo za trening skup,30% podataka za test skup
            int n = data.Length;
            int nTrain = n / 100 * 70;
            double[][] trainData = new double[nTrain][];
            double[] trainLabels = new double[nTrain];
            double[][] testData = new double[n - nTrain][];
            double[] testLabels = new double[n-nTrain];
            Helper.SplitTrainTestData(data, allLabels, trainData, testData, trainLabels, testLabels, n);

            //iz trening i test skupa izvlacimo poslednju kolonu jer se u njoj nalaze labele(klase) kojima observacije pripadaju

            int[] trainLabelsI = Helper.ConvertDoubleToInt(trainLabels);
            int[] testLabelsI = Helper.ConvertDoubleToInt(testLabels);

            //Kreiranje objekta klase MultinomialLogisticLearning pomocu kojeg cemo uciti model
            MultinomialLogisticLearning<Accord.Math.Optimization.GradientDescent> lrn = new MultinomialLogisticLearning<Accord.Math.Optimization.GradientDescent>();
		
			//Metoda Learn vraca istrenirani model 
            MultinomialLogisticRegression mlr = lrn.Learn(trainData, trainLabelsI);
			//Nakon treniranja vrsimo predikciju nad trening skupom i merimo preciznost
            int[] decisions = mlr.Decide(trainData);
            double trainAcc = Helper.CalculateAccuracy(decisions, trainLabelsI);
            int[] testDecisions = mlr.Decide(testData);

            double testAcc = Helper.CalculateAccuracy(testDecisions,testLabelsI);
            Console.WriteLine("Preciznost na trening skupu je  {0}", trainAcc * 100);
			//Vrsimo predikciju nad test skupom i merimo preciznost
            Console.WriteLine("Preciznost na test skupu je  {0}", testAcc * 100);





        }
    }
}

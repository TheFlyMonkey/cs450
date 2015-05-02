/*
 * I worked with Jordan Balls on this assignment. We complement eachother :)
 */
package weka;

import java.util.TreeMap;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class KNearestNeighbor extends Classifier{
    
    Instances trainningSet;
    int k = 10;
    
   
    @Override
    public void buildClassifier(Instances _trainningSet) throws Exception 
    {
        trainningSet = new Instances(_trainningSet);
    }

    @Override
    public double classifyInstance(Instance newInstance) throws Exception 
    {
        //This will just go through the map and see which number is the lowest
        //perhaps it will sort the map so that we can always just pick the K number
        //of items from the top and compare them to eachother and then choose the 
        //nearest one of those
        TreeMap<Double, Double> mapDistances = new TreeMap<>();
        for (int i = 0; i < trainningSet.numInstances(); i++)
        {
            Instance tmpInstance = trainningSet.instance(i);
            mapDistances.put(getDistance(tmpInstance, newInstance),tmpInstance.classValue());
        }
        
        
        return findNearestGroup(mapDistances, k);
    }
    
    //pass it the mapp and find the k smallest distances and classify it as those neighbors
    public double findNearestGroup(TreeMap<Double, Double> mapOfDistances, int k)
    {
         
        int[] tally = new int[trainningSet.numClasses()];
        for (int i = 0; i < k; i ++)
        {
            double index = mapOfDistances.values().iterator().next();
            tally[(int) index]++; 
        }
        
        int tallyOfNearestNeighbors = 0;
        int nearestNeighbor = 0;
        for (int i =0; i < tally.length; i++)
        {
            tallyOfNearestNeighbors = tally[i];
            if (tallyOfNearestNeighbors > tally[nearestNeighbor])
            {
                nearestNeighbor = i;
            }
        }
        
         return (double)nearestNeighbor;
    }
    
    public double getDistance(Instance trainningInstance, Instance newInstance)
    {
        //calulate the distance between the trainnningDataSet the newDataPoint(s)
        //create a map and put all the trainninDataSet instances in the map along
        //with their distances from the newDataPoint. A map because then we have all
        //the tranningSet still and a way to find immediately how close each one was
        //to the newDataPoint
        double distance = 0;
        double finalValue = 0;
        
    
        for (int i = 0; i < trainningInstance.numAttributes() - 1; i++)
        { 
            if (trainningInstance.attribute(i).isNumeric()) 
            {
                distance = Math.abs(trainningInstance.value(i) - newInstance.value(i));
            } 
            else 
            {
                if (!trainningInstance.stringValue(i).equals(newInstance.stringValue(i))) 
                {
                    distance = 1;
                }
            }
            distance = Math.abs(trainningInstance.value(i) - newInstance.value(i));
            finalValue += Math.pow(distance, (trainningInstance.numAttributes() - 1));
        }
        
        
        return Math.pow(finalValue, 1.0/(trainningInstance.numAttributes() - 1));
    }
}

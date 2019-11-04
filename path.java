import java.util.*;
import java.lang.Math; 

public class path {
  
    public static void main(String[] args) {  
        
        Solution(4, 10,10);
        Solution(8,12,12);   
    }
    
    
    static void Solution(int d, int step, int grid)
    {
        ///find the number
        Map<List<Integer>, long[]> map=new HashMap<>();
        Map<List<Integer>,Integer> count=new HashMap<>();
        
        System.out.println(" d "+d+" step "+step+" grid "+grid);
        ///find the path from origin 
        System.out.println(" the number of path from origin "+find2(new int[d], step, d, step, grid, map));
        
        ////find the distribution 
        distribution(0,new int[d], d, step, grid, map,count);
        int total=0;
        double sum=0, res=0;
        double min=Long.MAX_VALUE, max=0;
        for(List<Integer> key:count.keySet())
        {
            sum+=(map.get(key)[step]*count.get(key));
            total+=count.get(key);
            
        }
        double mean=sum/total;
//         System.out.println("mean "+mean+" total "+total);
        for(List<Integer> key:count.keySet())
        {
            long value=map.get(key)[step];
            sum+=(value-mean)*(value-mean)*count.get(key);
            min=Math.min(min, value);
            max=Math.max(max, value);
        }
        double std=Math.sqrt(sum*2/(total*2-1));
        res=std/mean;
        double maxmin=max/min;
        System.out.println("The ratio of max and min is "+maxmin);
        System.out.println(" ratio of std and mean is "+res);
        
    }
    
    
    static void distribution(int pos, int[] point, int d, int step, int grid, Map<List<Integer>, long[]> map, Map<List<Integer>,Integer> count )
    {
        if(pos==d)
        {
            List<Integer> list=new ArrayList<>();
            for(int p:point)
                list.add(p);
            Collections.sort(list);
            if(count.containsKey(list))
                count.put(list, count.get(list)+1);
            else
            {
                count.put(list, 1);
                find2(point,step, d, step, grid, map);
            }
//             cnt++;
            return;
        }
        for(int i=0; i<grid/2; i++)
        {
            point[pos]=i;
            distribution(pos+1, point, d, step, grid, map, count);
            point[pos]=0;
        }
        
    }
   
    static long find2(int[] point, int pos, int d, int step, int grid, Map<List<Integer>, long[]> map)
    {
        if(pos==0)
            return 1;
        List<Integer> list=new ArrayList<>();
        for(int p:point)
            list.add(p);
        Collections.sort(list);
        if(map.containsKey(list)&&map.get(list)[pos]!=0)
            return map.get(list)[pos];
        long res=0;
        for(int i=0; i<point.length; i++)
        {
            point[i]+=1;
            if(point[i]<grid&&point[i]>=0)
                res+=find2(point, pos-1, d, step,grid,map);
            point[i]-=2;
            if(point[i]<grid&&point[i]>=0)
                res+=find2(point, pos-1,d, step, grid,map);
            point[i]+=1;
        }
        
        long[] arr=map.getOrDefault(list, new long[step+1]);
        arr[pos]=res;
        map.put(list,arr);
        return res;
        
    }
    
    
    
    /////////////// debug 
    
    static long find(int step, int[] d, int n)
    {
        if(step==0)
            return 1;
        int res=0;
        for(int i=0; i<d.length; i++)
        {
            d[i]+=1;
            if(d[i]<n&&d[i]>=0)
                res+=find(step-1, d, n);
            d[i]-=2;
            if(d[i]<n&&d[i]>=0)
                res+=find(step-1,d,n);
            d[i]+=1;
        }
        return res;
    }
  static void plist(List<Integer> list)
    {
        for(int l:list)
            System.out.print(l+" ");
        System.out.println();
    } 
    
  static void print(int[] list)
    {
        for(int l:list)
            System.out.print(l+" ");
        System.out.println();
    } 
    
}
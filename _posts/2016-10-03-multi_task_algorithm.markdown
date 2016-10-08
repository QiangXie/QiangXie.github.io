---
layout: post
title: "任务执行顺序题的解决"
subtitle: "排序算法小结"
author: "Johnny"
date: 2016-10-04 12:15:09
header-img: "img/algorithm.png"
tags: 
    - algorithm
    - C++
---


在51nod学习贪心算法入门课程，课程最后一道题如下：

**任务执行顺序**

有N个任务需要执行，第i个任务计算时占R[i]个空间，而后会释放一部分，最后储存计算结果需要占据O[i]个空间（O[i] < R[i]）。例如：执行需要5个空间，最后储存需要2个空间。给出N个任务执行和存储所需的空间，问执行所有任务最少需要多少空间。

最佳策略是按照R[i] – O[i]的不增顺序执行，我按照这个策略写了程序，可是怎么都ac不过去，我就下载了测试样例看了看，一看就知道问题出在哪了，测试样例足足十万个输入数据。而我排序用的是很简单的选择排序，时间肯定超时，过了才是出了鬼了呢。我把代码复制到本地，输入数据运行之后，果然过了N分钟之后结果才出来，虽然结果是对的。

选择排序不行那只好换别的排序算法了，虽然我看过别的排序算法，但真正自己实现这些排序算法还真没实现过，平常遇到排序一般数据量比较小，通常都是用选择排序搞定，今天正好借此机会把排序算法好好学习一下。
#1.冒泡排序#
首先是冒泡排序，冒泡排序非常暴力，算法就是不断地扫描数组，看相邻的两个元素符不符合a[i] < a[i+1]的原则，如果不符合，就交换两个相邻元素，直到所有元素都符合这一规律为止。在排序的过程中，小的元素像水泡一样不断上浮，冒泡排序由此得名，算法复杂度O（N）（我这里都是用的vector而不是数组）

    void BubbleSort(vector<int> &nums)
    {
    	bool sortedFlag = true;
    	for (int i = 0; i < nums.size(); ++i)
    	{
    		for (int j = nums.size() - 1; j > i; --j)
    		{
    			if (nums[j] < nums[j - 1])
    			{
    				int temp = nums[j];
    				nums[j] = nums[j - 1];
    				nums[j - 1] = temp;
    				sortedFlag = false;
    			}
    		}
    		if (sortedFlag)
    			break;
    	}
    }

# 2.选择排序 #
选择排序是我平常最常用的排序算法，因为它写起来很简单。选择排序的思想是这样的：对数组从前往后扫描，每次扫描的起点不断往后移动，每趟扫描都找到这次扫描得到的最小元素，然后跟起点进行交换，这样经过N趟选择排序之后就变为有序的。选择排序代码如下：

    void SelectSort(vector<int> &nums)
    {
    	int min_index;
    	int temp;
    	for (int i = 0; i < nums.size(); ++i)
    	{
    		min_index = i;
    		for (int j = nums.size() - 1; j > i; --j)
    		{
    			if (nums[min_index] > nums[j])
    			{
    				min_index = j;
    			}
    		}
    		temp = nums[min_index];
    		nums[min_index] = nums[i];
    		nums[i] = temp;
    	}
    }

# 3.插入排序 #
插入排序是这样一种思想：每趟排序我们都把前面扫描过的数排好。实现策略是：第m次扫描前前m-1个元素已经排好了，我们只需要在前面m个数中给第m个元素找到一个合适的位置并把它放到合适的位置就行。具体做法是：让第m个元素从后往前依次和第m-1、m-2、m-3……个元素进行比较，如果这个第m个元素小于和他比较的元素，我们就交换这两个元素，直到这个元素不再小于前面的元素为止。经过n次这样的比较之后就能把元素都排好序，代码如下：

    void InsertSort(vector<int> &nums)
    {
    	int temp;
    	for (int i = 1; i < nums.size(); i++)
    	{
    		for (int j = i - 1; j >= 0&&nums[j+1] < nums[j]; j--)
    		{
    			temp = nums[j + 1];
    			nums[j + 1] = nums[j];
    			nums[j] = temp;
    		}
    	}
    }

#4.希尔排序#

先解释一个概念，**逆序**。成员数组的一个逆序（inversion）是指数组中具有性质i<j但A[i]>A[j]的序偶。我们排序的过程就是消除数组中的逆序的过程，上面三种算法在排序过程中每次比较都只消除一个逆序。如果我们每次比较不止消除了一个逆序，那所要进行的比较不就少了吗？事实确实如此，希尔排序和后面的归并排序、快速排序算法总的来说都是这么个思想。

希尔排序是Donald Shell发明的，它又叫做缩小增量排序算法。顾名思义，这个算法采用逐渐减小比较时扫描所用的步长，直到最后一趟排序用步长为1进行比较之后，数组也就排好了。代码如下：

    void ShellSort(vector<int> & nums)
    {
    	int step = 1;
    	while (step < nums.size())
    	{
    		step = step * 3 + 1;
    	}
    	while (step > 1)
    	{
    		step = step / 3 + 1;
    		for (int i = 0; i < step; i++)
    		{
    			int min_index;
    			int temp;
    			for (int j = i; j < nums.size(); j += step)
    			{
    				min_index = j;
    				for (int k = j; k < nums.size(); k += step)
    				{
    					if (nums[k] < nums[min_index])
    					{
    						min_index = k;
    					}
    				}
    				temp = nums[j];
    				nums[j] = nums[min_index];
    				nums[min_index] = temp;
    			}
    		}
    	}
    }
 
# 5.归并排序 #

归并采用分而治之的思想，它递归地不断把一个数组分成两部分，然后把经过排序后的两个表合并起来。合并的时候是这样做的：数组A和B已经排好序了，我们合并开始的时候用两个指针分别指向A和B的第一个元素，如果指向A的指针所指向的元素比较小就把A放到归并后的第一个元素，A的指针往后移动一位，反之对B也是，直到把两组元素全部合并为止。代码如下：

    void MergeSort(vector<int> & nums)
    {
    	if (nums.size() <= 1)
    	{
    		return;
    	}
    	else
    	{
    		vector<int> temp1, temp2;
    		for (int i = 0; i < nums.size() / 2; ++i)
    		{
    			temp1.push_back(nums[i]);
    		}
    		for (int i = nums.size() / 2; i < nums.size(); ++i)
    		{
    			temp2.push_back(nums[i]);
    		}
    		MergeSort(temp1);
    		MergeSort(temp2);
    		int index_1 = 0;
    		int index_2 = 0;
    		int index = 0;
    		while (index_1 < temp1.size() && index_2 < temp2.size())
    		{
    			if (temp1[index_1] < temp2[index_2])
    			{
    				nums[index] = temp1[index_1];
    				index_1++;
    				index++;
    			}
    			else
    			{
    				nums[index] = temp2[index_2];
    				index_2++;
    				index++;
    			}
    		}
    		if (index_1 == temp1.size())
    		{
    			while (index_2 < temp2.size())
    			{
    				nums[index] = temp2[index_2];
    				index_2++;
    				index++;
    			}
    		}
    		else
    		{
    			while (index_1 < temp1.size())
    			{
    				nums[index] = temp1[index_1];
    				index_1++;
    				index++;
    			}
    		}
    	}
    }

# 6.快速排序 #

快速排序顾名思义是已知的最快的排序算法，它的平均时间复杂度是O（NlogN）。它不断地把数组分为两个部分，和希尔排序不同的地方是它不是按照元素个数进行划分，而是找到一个枢纽元素，然后按照和枢纽元素进行比较把元素分为比枢纽元小的和比枢纽元大的两部分。然后再分别对分成的两个数组进行quick sort。它比归并排序算法好的地方在于它不需要临时的内存用来存放临时数组，比归并排序节省内存。快排算法的代码如下（为了保持接口的一致性写了一个用于递归调用的函数`void QuickSort_(vector<int> &nums, int start, int end)`）：

    void QuickSort_(vector<int> &nums, int start, int end)
    {
    	int size = end - start + 1;
    	int pivot_index;
    	if (size <= 1)
    	{
    		return;
    	}
    	else
    	{
    		if ((nums[start] >= nums[end] && nums[start] <= nums[(start + end) / 2]) || (nums[start] <= nums[end] && nums[start] >= nums[(start + end) / 2]))
    		{
    			pivot_index = start;
    		}
    		else if ((nums[end] >= nums[start] && nums[end] <= nums[(start + end) / 2]) || (nums[end] <= nums[start] && nums[end] >= nums[(start + end) / 2]))
    		{
    			pivot_index = end;
    		}
    		else if ((nums[(start + end) / 2] >= nums[start] && nums[(start + end) / 2] <= nums[end]) || (nums[(start + end) / 2] <= nums[start] && nums[(start + end) / 2] >= nums[end]))
    		{
    			pivot_index = (start + end) / 2;
    		}
    		int temp = nums[pivot_index];
    		nums[pivot_index] = nums[end];
    		nums[end] = temp;
    		int i = start;
    		int j = end - 1;
    		while (i <= j)
    		{
    			if (nums[i] >= nums[end] && nums[j] <= nums[end])
    			{
    				int temp_ = nums[i];
    				nums[i] = nums[j];
    				nums[j] = temp_;
    				i++;
    				j--;
    			}
    			else if (nums[i] < nums[end] && nums[j] > nums[end])
    			{
    				i++;
    				j--;
    			}
    			else if (nums[i] >= nums[end] && nums[j] > nums[end])
    			{
    				j--;
    			}
    			else if (nums[i] < nums[end] && nums[j] <= nums[end])
    			{
    				i++;
    			}
    		}
    		temp = nums[i];
    		nums[i] = nums[end];
    		nums[end] = temp;
    		QuickSort_(nums, start,i - 1 );
    		QuickSort_(nums, i + 1, end);
    	}
    
    }
    
    void QuickSort(vector<int> &nums)
    {
    	int counter = nums.size();
    	QuickSort_(nums, 0, counter - 1);
    }

#7.任务顺序算法 #

回到前面碰到的关于任务顺序的题，采用归并排序算法后顺利AC。代码如下：

    #include <iostream>
    #include <vector>
    
    using namespace std;
    
    struct task
    {
    	int occupied_memory;
    	int storage_memory;
    };
    
    typedef task* pttask;
    
    void MergeSort(vector<pttask> & nums)
    {
    	if (nums.size() <= 1)
    	{
    		return;
    	}
    	else
    	{
    		vector<pttask> temp1, temp2;
    		for (int i = 0; i < nums.size() / 2; ++i)
    		{
    			temp1.push_back(nums[i]);
    		}
    		for (int i = nums.size() / 2; i < nums.size(); ++i)
    		{
    			temp2.push_back(nums[i]);
    		}
    		MergeSort(temp1);
    		MergeSort(temp2);
    		int index_1 = 0;
    		int index_2 = 0;
    		int index = 0;
    		while (index_1 < temp1.size() && index_2 < temp2.size())
    		{
    			if (temp1[index_1]->occupied_memory - temp1[index_1]->storage_memory >= temp2[index_2]->occupied_memory - temp2[index_2]->storage_memory)
    			{
    				nums[index] = temp1[index_1];
    				index_1++;
    				index++;
    			}
    			else
    			{
    				nums[index] = temp2[index_2];
    				index_2++;
    				index++;
    			}
    		}
    		if (index_1 == temp1.size())
    		{
    			while (index_2 < temp2.size())
    			{
    				nums[index] = temp2[index_2];
    				index_2++;
    				index++;
    			}
    		}
    		else
    		{
    			while (index_1 < temp1.size())
    			{
    				nums[index] = temp1[index_1];
    				index_1++;
    				index++;
    			}
    		}
    	}
    }
    
    int main()
    {
    	int task_num;
    	int ans = 0;
    	vector<pttask> tasks;
    	cin >> task_num;
    	int task_num_ = task_num;
    	pttask temp;
    	while (task_num_--)
    	{
    		temp = new task;
    		cin >> temp->occupied_memory >> temp->storage_memory;
    		tasks.push_back(temp);
    	}
    	MergeSort(tasks);
    	int temp_time = 0;
    	for (int i = 0; i < tasks.size(); ++i)
    	{
    		temp_time += tasks[i]->occupied_memory;
    		if (temp_time > ans)
    		{
    			ans = temp_time;
    		}
    		temp_time -= tasks[i]->occupied_memory - tasks[i]->storage_memory;
    	}
    	cout << ans << endl;
    
    	return 0;
    }


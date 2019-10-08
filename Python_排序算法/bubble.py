def bubble_sorted(iterable):
    '''
    冒泡排序算法的运作如下：
        1.比较相邻的元素。如果第一个比第二个大，就交换他们两个。
        2.对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
        3.针对所有的元素重复以上的步骤，除了最后一个。
        4.持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
    '''
    for i in range(len(iterable)):
        for j in range(i+1,len(iterable)):
            if iterable[i] > iterable[j]:
                iterable[i] , iterable[j] = iterable[j] , iterable[i]
    return iterable

if __name__ == "__main__":
    testlist = [27, 33, 28, 4, 2, 26, 13, 35, 8, 14]
    print('sorted:', bubble_sorted(testlist))
    # sorted: [2, 4, 8, 13, 14, 26, 27, 28, 33, 35]
def shell_sort(list):
    '''
    希尔排序，也称递减增量排序算法，是插入排序的一种更高效的改进版本
    1.插入排序在对几乎已经排好序的数据操作时，效率高，即可以达到线性排序的效率
    2.但插入排序一般来说是低效的，因为插入排序每次只能将数据移动一位
    '''
    n = len(list)
    gap = n // 2
    while gap > 0:
        for i in range(gap,n):
            # 每个步長進行插入排序
            temp = list[i]
            j = i
            # 插入排序
            while j >= gap and list[j-gap] > temp:
                list[j] = list[j-gap]
                j -= gap
            list[j] = temp
        # 得到新的步长
        gap = gap // 2
    return list

if __name__ == '__main__':
    testlist = [17, 23, 20, 14, 12, 25, 1, 20, 81, 14, 11, 12]
    print(shell_sort(testlist))

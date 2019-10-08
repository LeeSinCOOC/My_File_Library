def selection_sort(arr):
    '''
    1.首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置
    2.然后，再从剩余未排序元素中继续寻找最小（大）元素
    3.然后放到已排序序列的末尾。
    4.以此类推，直到所有元素均排序完毕。
    '''
    for i in range(len(arr)-1):
        minIndex=i
        for j in range(i+1,len(arr)):
            if arr[minIndex]>arr[j]:
                minIndex=j
        if i==minIndex:
            pass
        else:
            arr[i],arr[minIndex]=arr[minIndex],arr[i]
    return arr


if __name__ == '__main__':
    testlist = [17, 23, 20, 14, 12, 25, 1, 20, 81, 14, 11, 12]
    print(selection_sort(testlist))

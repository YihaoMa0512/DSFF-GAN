import os
from multiprocessing import Pool
import numpy as np
from PIL import Image
from sewar.full_ref import ssim, msssim, psnr
import time
from tqdm import tqdm
import statistics
import xlsxwriter


def jiegou(srcpath):
    output_ssim = []
    output_psnr = []
    output_ccs = []
    ssim_sum = 0
    ccs_sum = 0
    psnr_sum = 0
    results = []
    for i in tqdm(range(srcpath.shape[0])):
        img1 = Image.open(srcpath[i][0])
        img2 = Image.open(srcpath[i][1])
        img1 = np.array(img1)
        img2 = np.array(img2)
        degree_psnr = psnr(img1, img2)
        degree_ssim, degree_ccs = ssim(img1, img2, ws=8)
        output_psnr.append(degree_psnr)
        output_ssim.append(degree_ssim)
        output_ccs.append(degree_ccs)
        ssim_sum = ssim_sum + degree_ssim
        ccs_sum = ccs_sum + degree_ccs
        psnr_sum = psnr_sum + degree_psnr

        file_name = os.path.basename(srcpath[i][0])
        results.append([file_name, degree_ssim, degree_ccs, degree_psnr])

    return ssim_sum, ccs_sum, psnr_sum, output_ssim, output_ccs, output_psnr, results


if __name__ == '__main__':
    time1 = time.time()
    total_sums_ssim = 0
    total_sums_ccs = 0
    total_sums_psnr = 0
    list_ssim = []
    list_ccs = []
    list_psnr = []
    output_results = []

    num_workers = 32
    mode='dsff2_2_2'
    real_path='../datasets2/test/B/'
    fake_path='../results/data_2/'+mode+'/'
    a = [real_path + str(elem) for elem in sorted(os.listdir(real_path))]
    b = [fake_path + str(elem) for elem in sorted(os.listdir(fake_path))]
    data_list = [[lst1_elem, lst2_elem] for (lst1_elem, lst2_elem) in zip(a, b)]
    l = len(data_list)
    print()
    # 使用多进程分配任务，每个工作进程处理部分数据
    with Pool(num_workers) as pool:
        # 将图像列表分成多个部分
        file_lists = np.array_split(data_list, num_workers)

        # 启动多个进程处理图像数据
        results = pool.map(jiegou, file_lists)

        for result in results:
            total_sums_ssim += result[0] * 100
            total_sums_ccs += result[1] * 100
            total_sums_psnr += result[2]
            list_ssim += result[3]
            list_ccs += result[4]
            list_psnr += result[5]
            output_results.extend(result[6])

        output_results.sort(key=lambda x: x[0])  # Sort the output_results based on file names
        # Create a new Excel file and add a worksheet
        workbook = xlsxwriter.Workbook('data2_dsff.xlsx')
        worksheet = workbook.add_worksheet()

        # Write the results to the Excel file
        worksheet.write_row(0, 0, ['File Name', 'SSIM', 'CCS', 'PSNR'])
        for i, row in enumerate(output_results, start=1):
            worksheet.write_row(i, 0, row)

        workbook.close()

        print('ssim=', total_sums_ssim / l, '+-', statistics.stdev(list_ssim), '\n'
                                                                               'ccs=', total_sums_ccs / l, '+-',
              statistics.stdev(list_ccs), '\n'
                                          'psnr=', total_sums_psnr / l, '+-', statistics.stdev(list_psnr))
    time2 = time.time()
    print('总共耗时：' + str(time2 - time1) + 's')

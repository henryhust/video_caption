# coding:utf-8
in_file = '..\\data\\ch_video_train_seg.csv'
out_file = 'train_data_ch.csv'
file = open(in_file, 'r',encoding="utf8")
out = open(out_file, 'w',encoding="utf8")
i = 0
write_flag = True
pre_file  = ''
for line in file:
    i+=1
    line = line.strip()
    if line == '':
        continue
    line_l = line.split(',')
    try:
        if line_l[6] == 'Chinese':
            if write_flag == False:
                continue
            if line_l[0] == pre_file:
                continue
            pre_file = line_l[0]
            out.write(line+'\n')
            out.write('\n')
            write_flag = False
        else:
            write_flag = True

    except:
        print('debug')

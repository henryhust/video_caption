
def get_word_vec_dict():
    model_text_path = 'word_vec_video_ch.txt'
    model_file = open(model_text_path, 'r')
    i = -1
    words = 0
    vec_dim = 0
    word_vec_dict = {}

    for line in model_file:
        i += 1
        if i == 0:
            line_l = line.strip().split(' ')
            words = int(line_l[0])
            vec_dim = int(line_l[1])
            continue
        line_l = line.strip().split(' ')
        if i > words:
            break

        word_vec_dict[line_l[0]] = list(map(lambda x: float(x), line_l[1:]))


    print('load word vector success')
    return  word_vec_dict



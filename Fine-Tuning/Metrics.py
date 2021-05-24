import numpy as np
np.random.seed(0)
#This code based on https://github.com/chunyuanY/Dialogue metric

class Metrics(object):

    def __init__(self, score_file_path:str):
        super(Metrics, self).__init__()
        self.score_file_path = score_file_path
        self.segment = 10 #It depend on positive negative ratio 1:1 or 1:10

    def __read_socre_file(self, score_file_path):
        sessions = []
        one_sess = []
        with open(score_file_path, 'r') as infile:
            i = 0
            for line in infile.readlines():
                i += 1
                tokens = line.strip().split('\t')
                one_sess.append((float(tokens[0]), int(tokens[1])))
                if i % self.segment == 0:
                    one_sess_tmp = np.array(one_sess)
                    if one_sess_tmp[:, 1].sum() > 0:
                        sessions.append(one_sess)
                    one_sess = []
        return sessions


    def __mean_average_precision(self, sort_data):
        #to do
        count_1 = 0
        sum_precision = 0
        for index in range(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index+1)
        return sum_precision / count_1


    def __mean_reciprocal_rank(self, sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

    def __precision_at_position_1(self, sort_data):
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0

    def __recall_at_position_k_in_10(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)


    def evaluation_one_session(self, data):
        
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = self.__mean_average_precision(sort_data)
        m_r_r = self.__mean_reciprocal_rank(sort_data)
        p_1   = self.__precision_at_position_1(sort_data)
        r_1   = self.__recall_at_position_k_in_10(sort_data, 1)
        r_2   = self.__recall_at_position_k_in_10(sort_data, 2)
        r_5   = self.__recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5


    def evaluate_all_metrics(self):
        sum_m_a_p = 0
        sum_m_r_r = 0
        sum_p_1 = 0
        sum_r_1 = 0
        sum_r_2 = 0
        sum_r_5 = 0

        sessions = self.__read_socre_file(self.score_file_path)
        total_s = len(sessions)
        for session in sessions:
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

        return (sum_m_a_p/total_s,
                sum_m_r_r/total_s,
                  sum_p_1/total_s,
                  sum_r_1/total_s,
                  sum_r_2/total_s,
                  sum_r_5/total_s)


if __name__ == '__main__':
    metric = Metrics('ubuntu_score_file0.txt')
    result = metric.evaluate_all_metrics()
    for r in result:
        print(r)










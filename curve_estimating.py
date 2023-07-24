from matplotlib import pyplot as plt
import numpy
import loadimage

accuracy = 0.001
def f(x):
    return numpy.log(x+1)
    return numpy.sin(6*x)
    return x**4 - 3 * x**3 + 2 * x**2 + 0.5 * x

__range = numpy.arange(0, 3, accuracy)
__fx = loadimage.read('linetest.png')[0, :, 0]
#__fx = f(__range) + numpy.random.random(len(__range)) * 0.005
plt.plot(__range, __fx)
plt.show()
#input()

def random_noise_func(x):
    #return f(x)
    return __fx[numpy.array(x*1/accuracy, dtype = 'int32')]



def curve_function(x, corr_attributions, corr_scale = 1, base_func = random_noise_func):
    return base_func(x) - correction_function(x, corr_attributions) * corr_scale

def expir_function(x_start, x_end, corr_attributions = [], step = accuracy, base_func = random_noise_func):
    _range = numpy.arange(x_start, x_end, step)
    _result = curve_function(_range, corr_attributions, base_func = base_func)
    
    __top = sum((_range - (x_start + x_end)/2)*(_result - numpy.mean(_result)))
    __bottom = sum((_range - (x_start + x_end)/2)*(_range - (x_start + x_end)/2))
    
    return __top / __bottom

def correction_function(x, corr_attributions):
    #  修正函数参数的格式 列表，每一项为 范围下界，范围上界，斜率
    #  单变量情况，比较简单
    if str(type(x)) != "<class 'numpy.ndarray'>":
        delta = 0
        for attr in corr_attributions:
            _start, _end, _k, _b = attr
            if _start < x <= _end:
                delta += _k * (x - _start) + _b
        return delta

    #  numpy.array 时
    length = len(x)
    delta = numpy.zeros((length,))
    for attr in corr_attributions:
        _start, _end, _k ,_b = attr
        target_range = numpy.array((numpy.array(_start <= x, dtype = 'int8') + numpy.array(x < _end, dtype = 'int8')) == 2, dtype = 'int8')  #  得到可用区间
        range_with_k = target_range * (_k * (x - _start) + _b)
        delta = delta + range_with_k
    return delta

def calculate_func_avg(x_start, x_end, corr_attributions = [], step = accuracy, base_func = random_noise_func):
    _range = numpy.arange(x_start, x_end, step)
    return sum(curve_function(_range, corr_attributions, base_func = base_func)) / len(_range)

def calculate_error_result(x_start, x_end, corr_attributions = [], step = accuracy, base_func = random_noise_func):
    _range = numpy.arange(x_start, x_end, step)
    return _range, curve_function(_range, corr_attributions, base_func = base_func)
    
def calculate_corr_result(x_start, x_end, corr_attributions = [], step = accuracy, base_func = random_noise_func):
    _range = numpy.arange(x_start, x_end, step)
    return _range, correction_function(_range, corr_attributions)


if __name__ == '__main__':
    left, right = (0 ,3)
    corr_attributions = []
    lasttime_error = 10000000000
    for generation in range(10):
        pattern_amount = 2 ** generation
        start_points = numpy.arange(left, right, (right - left) / pattern_amount)
        end_step = (right - left) / pattern_amount
        
        for patt in range(pattern_amount):
            st, ed = (start_points[patt], start_points[patt] + end_step)
            avg = calculate_func_avg(st, ed, corr_attributions = corr_attributions)
            k = expir_function(st, ed, corr_attributions = corr_attributions)   #(curve_function(ed, corr_attributions) - curve_function(st, corr_attributions)) / (end_step * 2)
            b = avg# + k * end_step
            corr_attributions.append((st, ed, k, b))
        
        result = calculate_corr_result(left, right, corr_attributions)
        print('mean:', numpy.mean(result[1]), '\tvar:', numpy.var(result[1]), '\tmax_error:', numpy.max(abs(result[1])))
        if numpy.max(abs(result[1])) > lasttime_error and generation > 3 and False:
            plt.plot(result[0], result[1])
            plt.show()
            
            plt.plot(result[0], __fx - result[1])
            plt.show()
            break
        plt.plot(__range, __fx)
        plt.plot(result[0], result[1])
        plt.show()
        lasttime_error = numpy.max(abs(result[1]))
    
        
      
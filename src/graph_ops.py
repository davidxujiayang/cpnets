import tensorflow as tf

        
def weight_mul(x, weight):
    if tf.rank(weight)==1:
        return tf.einsum('ijk,j->ijk', x, weight)
    elif tf.rank(weight)==2:
        return tf.einsum('ijk,ij->ijk', x, weight)
    else:
        return x*weight

def gauss_linear_gradient(x, index_i, index_j, N, di, dj, a, vec, v_inv):
    xi = tf.gather(x, index_i, axis=1)
    xj = tf.gather(x, index_j, axis=1)
    f = di/(di+dj)
    # xf = (xi+f*(xj-xi))*a
    xf = weight_mul(xi+weight_mul(xj-xi, f), a)
    xf_x = weight_mul(xf, vec[...,0])
    xf_y = weight_mul(xf, vec[...,1])
    xfi_x = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(xf_x, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])
    xfj_x = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(xf_x, perm=[1, 0, 2]), index_j, N), perm=[1, 0, 2])
    xfi_y = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(xf_y, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])
    xfj_y = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(xf_y, perm=[1, 0, 2]), index_j, N), perm=[1, 0, 2])
    gradient_x = weight_mul(xfi_x-xfj_x, v_inv)
    gradient_y = weight_mul(xfi_y-xfj_y, v_inv)
    return gradient_x, gradient_y

def sum_edge(e, index_i, index_j, N, neighbor=False, weight=None):
    if weight is not None:
        e = tf.math.multiply(e, weight)
    if neighbor:
        return tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(e, perm=[1, 0, 2]), index_j, N), perm=[1, 0, 2])
    else:
        return tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(e, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])

def diff_neighbor(x, index_i, index_j, N, e=None, reverse=False, bidirectional=0, use_abs=False):
    if not reverse:
        messages = tf.gather(x, index_j, axis=1) + tf.gather(x, index_i, axis=1)
    else:
        messages = tf.gather(x, index_j, axis=1) - tf.gather(x, index_i, axis=1)
    
    if use_abs:
        messages = tf.math.abs(messages)

    if e is not None:
        messages = tf.math.multiply(messages, e)

    aggregated = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(messages, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])
    if bidirectional==1:
        aggregated += tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(messages, perm=[1, 0, 2]), index_j, N), perm=[1, 0, 2])
    elif bidirectional==-1:
        aggregated -= tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(messages, perm=[1, 0, 2]), index_j, N), perm=[1, 0, 2])

    return aggregated

def avg_neighbor(x, index_i, index_j, N, e=None, neighbor_only=False):
    messages = tf.gather(x, index_j, axis=1)
    aggregated = tf.transpose(tf.math.unsorted_segment_sum(tf.transpose(messages, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])
    if neighbor_only:
        return aggregated/8
    else:
        return (x+aggregated)/9

def min_neighbor(x, index_i, index_j, N):
    messages = tf.gather(x, index_j, axis=1)
    neighbor_min = tf.transpose(ops.scatter_min(tf.transpose(messages, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])
    return tf.math.minimum(x, neighbor_min)

def max_neighbor(x, index_i, index_j, N):
    messages = tf.gather(x, index_j, axis=1)
    neighbor_max = tf.transpose(ops.scatter_max(tf.transpose(messages, perm=[1, 0, 2]), index_i, N), perm=[1, 0, 2])
    return tf.math.maximum(x, neighbor_max)
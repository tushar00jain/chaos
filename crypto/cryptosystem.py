import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import struct



mpl.rcParams['lines.linewidth'] = .2


class TwoNodeSystem(object):
    def __init__(self, a1, a2, g1, g2, T1=1, T2=1, Transient_Iterations=None):
        '''Generic Hopfield Neural Network with Two nodes and no
        self connections

        initial conditions:
            - a1, a2: internal neuron decay
            - g1, g2: 
            - T1, T2:
        '''
        # Default Value that will not iterate
        self.Transient_Iterations = 2
        if Transient_Iterations:
            self.Transient_Iterations = Transient_Iterations
        self.a1 = a1
        self.a2 = a2
        self.g1 = g1
        self.g2 = g2
        self.T1 = T1
        self.T2 = T2

        self.prev_bit = None
        self.next_bit = None

        self.x_vals = None
        self.y_vals = None



    def iterate(self, N, x0=None, x1=None, x2=None, y0=None, y1=None, y2=None):
        if x0 ==None:
            x2 = self.x_vals[-1]
            x1 = self.x_vals[-2]
            x0 = self.x_vals[-3]

            y2 = self.y_vals[-1]
            y1 = self.y_vals[-2]
            y0 = self.y_vals[-3]
            

        self.x_vals = np.zeros(N+1, dtype=np.longdouble)
        self.y_vals = np.zeros(N+1, dtype=np.longdouble)


        self.x_vals[0] = x0
        self.x_vals[1] = x1
        self.x_vals[2] = x2

        self.y_vals[0] = y0
        self.y_vals[1] = y1
        self.y_vals[2] = y2

        # TODO Implement C Module to Run the Neural Netowork for Speed
        for n in range(3, N):
            self.x_vals[n+1] = self.a1*self.x_vals[n] + self.T1*self.g1(self.y_vals[n-2], dtype=np.longdouble)
            self.y_vals[n+1] = self.a2*self.y_vals[n] + self.T2*self.g2(self.x_vals[n-1], dtype=np.longdouble)
        return self.x_vals, self.y_vals

    def plot(self):
        plt.scatter(self.x_vals, self.y_vals)

def theta(t, Tau):
    '''The Threshold function for our value of t
    '''
    if t < Tau:
        output = 0
    else:
        output = 1
    return output

# TODO optimize with C
def bit_binary_rep(t, d, e, i):
    p_of_two = [ 0 ,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912,1073741824,2147483648,4294967296 ]

    bin_rep = 0
    for r in range(1, (2**i)):
        Tau = (e - d)*(r/(p_of_two[i])) + d
        bin_rep += ((-1)**(r - 1))*theta(t, Tau)
    return bin_rep

# TODO optimize with C
def Count(N, M, turn):
    INT_BITS=32
    def leftRotate(n, d):
        # In n<<d, last d bits are 0.
        # To put first 3 bits of n at
        # last, do bitwise or of n<<d
        # with n >>(INT_BITS - d)
        return (n << d)|(n >> (INT_BITS - d))

    # Function to right
    # rotate n by d bits
    def rightRotate(n, d):
    
        # In n>>d, first d bits are 0.
        # To put last 3 bits of at
        # first, do bitwise or of n>>d
        # with n <<(INT_BITS - d)
        return (n >> d)|(n << (INT_BITS - d)) & 0xFFFFFFFF
    # Convert N into hexadecimal number and
    # remove the initial zeros in it
    if(turn == 'R'):
        S = rightRotate(N, M)

    else:
        S = leftRotate(N, M)

    # Convert the rotated binary string
    # in decimal form, here 2 means in binary form.
    return S

def SegmentGenerator(Message, num_bytes=4):
    """
    takes a string message and breaks off num_bytes chunks
    """
    Message = Message.encode("utf-8")
    Message = np.frombuffer(Message, dtype="S1")

    if len(Message) % num_bytes == 0:
        output = np.split(Message, len(Message)/num_bytes)

    else:
        extra = len(Message) % num_bytes
        buff  = num_bytes - extra

        extra_bytes = [Message[-1*extra:]]
        Message = Message[:-1*extra]

        extra_bytes = np.append(extra_bytes, np.array([b' ']*buff, dtype="S1"))
        Message = np.append(Message,extra_bytes)
        output = np.split(Message, len(Message)/num_bytes)
    for val in output:
        yield b"".join([c for c in val])

def Compute_Binary_Sequence(rnn, first_time_flag=True):
        '''
        Creates AJ and DJ
        '''
        x_vals, y_vals = rnn.iterate(38)
        A_J = np.zeros(32, dtype=np.int8)
        D_J = np.zeros(5, dtype=np.int8)
        x_vals, y_vals = my_rnn_pos.iterate(N=38)
        if rnn.prev_bit != 1:
            for k in range(1, 33):
                    A_J[k-1] = bit_binary_rep(x_vals[k-1], -10, 0, i=4)
            q = 0
            for k in range(33, 38):
                D_J[q] = bit_binary_rep(x_vals[k-1], -10, 0, i=4)
                #print(str(k)+".", D_J[q])
                q += 1
                rnn.prev_bit = x_vals[-1]
        else:
            # Choose y
            for k in range(1, 33):
                    A_J[k-1] = bit_binary_rep(y_vals[k-1], -10, 0, i=4)
                #print(str(k)+".", A_J[k-1])
            q = 0
            for k in range(33, 38):
                D_J[q] = bit_binary_rep(y_vals[k-1], -10, 0, i=4)
                #print(str(k)+".", D_J[q])
                q += 1
                rnn.prev_bit = y_vals[-1]

        AJ = "0b"+"".join([str(x) for x in A_J])
        DJ = "0b"+''.join([str(x)for x in D_J])

        return AJ, DJ, A_J, D_J

def convert(data):
    with_flag = eval(bin(int(data, 16)))
    return int.to_bytes(with_flag, 4, 'big')

def unconvert(byte_data):
    bin_str = bin(int.from_bytes(byte_data, 'big'))
    flag = bin_str[-1]
    data = bin_str[:-1]
    return (hex(eval(data)))


if __name__=="__main__":

    b = -2
    print("Creating Hopfield Network ...")
    my_rnn_pos = TwoNodeSystem(a1=1/4, a2=3/4, g1=np.sin, g2=np.tanh, T1=1, T2=b)
    my_rnn_neg = TwoNodeSystem(a1=1/4, a2=3/4, g1=np.sin, g2=np.tanh, T1=1, T2=b)
    print("Successfully Created Network")

    # From The Paper ( Simple Example w/ Text)
    Message = input("Please Enter an encodable message:\n>> ")
    #STEP-1: Iterate the Nerual Network to Transient Pos
    x_N, y_N = my_rnn_pos.iterate(N=10001, x0=.01, x1=.01, x2=.01, y0=.01,y1=.01, y2=.01)
    x0 = x_N[-1:]

    #STEP-2: Break the Message into l=4 subsequences
    getChunk = SegmentGenerator(Message, num_bytes=4)

    final_message = ""
    for i, chunk in enumerate(getChunk):
        print("iteration {curr}".format(curr=i), end='\r')
        PJ = chunk

        #STEP-3: Compute the Binary Sequences Supplied by the Fourth bits

        #print("PJ:", PJ)
        #print("LEN:", len(PJ))

        AJ, DJ, A_J, D_J = Compute_Binary_Sequence(my_rnn_pos)
        a1 = int(AJ,2)
        #p1 = int("0b01100010011101010110011001100110",2)
        p1 = int.from_bytes(PJ, 'little')
        #print("PJ:", bin(p1))
        #print("AJ:", bin(a1))
        #print("DJ:", DJ)

        #print(p1)
        #print(a1)

        p1_prime = Count(p1, int(DJ,2), "L")
        a1_prime = Count(a1, int(DJ,2), "R")

        cypher = p1_prime ^ a1_prime

        #print("First Encryption:", cypher)
        #print("As Bytes:",bin(cypher))

        #print("Decrypting ...")
        P1 = a1_prime ^ cypher
        plaintext = Count(P1, int(DJ, 2), "R")

        #print(int.to_bytes(plaintext, 4, 'little').decode('utf-8'))

        x_N, y_N = my_rnn_pos.iterate(N=38)
        final_message += int.to_bytes(plaintext, 4, "little").decode('utf-8')
    print("\nCompletely Decrypted:",final_message)

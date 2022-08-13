from ast import Pass
from distutils.command.install_egg_info import safe_name
import streamlit as st
import quandl
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    st.header('3D US Yield Curve')
    ch=st.date_input('Select the starting date: ',value=datetime.strptime('2011-01-01','%Y-%m-%d'),max_value=datetime.strptime('2016-01-02','%Y-%m-%d'))
    data = quandl.get('USTREASURY/YIELD', returns='numpy', trim_start=ch)
    # Conversion
    header = []
    for name in data.dtype.names[1:]:
        maturity = float(name.split(" ")[0])
        if name.split(" ")[1] == 'Mo':
            maturity = maturity / 12
        header.append(maturity)

    x_data = []; y_data = []; z_data = []


    for dt in data.Date:
        dt_num = dates.date2num(dt)
        x_data.append([dt_num for i in range(len(data.dtype.names)-1)])


    for row in data:
        y_data.append(header)
        z_data.append(list(row.tolist()[1:]))

    x = np.array(x_data, dtype='f'); y = np.array(y_data, dtype='f'); z = np.array(z_data, dtype='f')


    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#00172B')
    ax.patch.set_facecolor('#00172B')
    ax.plot_surface(x, y, z, rstride=10, cstride=1, cmap='winter', vmin=np.nanmin(z), vmax=np.nanmax(z))
    ax.set_title('US Treasury Yield Curve',color='white')
    ax.set_ylabel('Maturity', color='white')
    ax.set_zlabel('Yield',color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    

    # SO question
    def format_date(x, pos=None):
        return dates.num2date(x).strftime('%Y-%m-%d')

    ax.w_xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    for tl in ax.w_xaxis.get_ticklabels():
        tl.set_ha('right')
        tl.set_rotation(15)

    st.pyplot(fig)

if __name__ == "__main__":
    main()

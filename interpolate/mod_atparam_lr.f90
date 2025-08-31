module mod_atparam_lr
    implicit none

    private
    public isc, ntrun, mtrun, ix, iy
    public nx, mx, mxnx, mx2, il, ntrun1, nxp, mxp, lmax

    integer, parameter :: isc = 1
    integer, parameter :: ntrun = 30, mtrun = 30, ix = 96, iy = 24
    integer, parameter :: nx = ntrun+2, mx = mtrun+1, mxnx = mx*nx, mx2 = 2*mx
    integer, parameter :: il = 2*iy, ntrun1 = ntrun+1
    integer, parameter :: nxp = nx+1 , mxp = isc*mtrun+1, lmax = mxp+nx-2
end module

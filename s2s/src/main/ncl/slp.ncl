f = addfile(".../slp.1980.nc", "r")
slp = f->slp    ; extract sea level pressure
printVarSummary(slp)

slpVN = slp(:,26:33,40:44) ; filter the Vietnam region [time, lat, lon]

wks = gsn_open_wks("png", "slp.vn.1980") ; write the plot to a PNG file

res = True
res@cnFillPalette   = "gui_default"
res@gsnContourZeroLineThicknessF = 2.
res@cnFillOn        = True

data = slpVN(0,:,:)/100
plot = gsn_csm_hov(wks, data, res)
# plot = gsn_csm_hov(wks, data, res)

gmt begin Figure_1_map png,pdf
    gmt set FONT_ANNOT_PRIMARY 14p,Helvetica,black
    gmt coast -R-4/10/48/58 -JB3/52/48/48/10c -Ba2f1 -G244/243/239 -S167/194/223 -N1/0.5p 
    gmt plot BCS.txt -Gyellow
    gmt plot DCS.txt -Glightgreen
    gmt plot Scheldt.txt -W1p,red
    
    echo S 0.3c r 0.3c yellow 0.5p 0.7c BCS > legend.txt
    echo S 0.3c r 0.3c lightgreen 0.5p 0.7c DCS >> legend.txt
    echo S 0.3c - 0.3c - 1p,red 0.7c SE >> legend.txt
    gmt legend -DjBR+o0.3c/0.3c -F+gwhite+p0.5p legend.txt

    gmt inset begin -DjTL+w1.5i+o0.4i/0.1i
        gmt set MAP_GRID_PEN_PRIMARY 25p,black,2_2
        gmt coast -JG3/53/1.5i -Rg -Bg -Wfaint -G244/243/239 -S167/194/223 -ETW+gred
        echo -4 48 10 58 | gmt plot -Sr+s -W1p,red
    gmt inset end
gmt end
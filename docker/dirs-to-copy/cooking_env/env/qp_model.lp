\ Model CBF_CLF_QP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 [ 2 x1_input_acceleration ^2 + 2 x2_input_acceleration ^2
   + 2 x3_input_acceleration ^2 + 2 relaxation_CLF ^2 ] / 2 
Subject To
 CBF_Constraint: - 3.528000116348267 x1_input_acceleration
   + 0.8999999523162838 x2_input_acceleration
   - 0.00200049877166752 x3_input_acceleration >= -3.224197184279508
 Relaxed_CLF_constraint: - 0.7959998846054077 x1_input_acceleration
   + 1.053000330924988 x2_input_acceleration
   - 0.0460002422332764 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.488000094890594 x1_input_acceleration
   + 0.8599999904632565 x2_input_acceleration
   + 0.0379994630813598 x3_input_acceleration >= -3.136797151187017
 Relaxed_CLF_constraint: - 0.7759998738765717 x1_input_acceleration
   + 1.033000349998474 x2_input_acceleration
   - 0.0260002613067627 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.448000073432922 x1_input_acceleration
   + 0.8200000286102291 x2_input_acceleration
   + 0.0779994249343872 x3_input_acceleration >= -3.051797115901077
 Relaxed_CLF_constraint: - 0.7559998631477356 x1_input_acceleration
   + 1.01300036907196 x2_input_acceleration
   - 0.00600028038024902 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.40800005197525 x1_input_acceleration
   + 0.7800000667572018 x2_input_acceleration
   + 0.1179993867874145 x3_input_acceleration >= -2.969197078421689
 Relaxed_CLF_constraint: - 0.7359998524188995 x1_input_acceleration
   + 0.9930003881454468 x2_input_acceleration
   + 0.0139997005462646 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.368000030517578 x1_input_acceleration
   + 0.7400001049041744 x2_input_acceleration
   + 0.0779994249343872 x3_input_acceleration >= -2.884277067778673
 Relaxed_CLF_constraint: - 0.7159998416900635 x1_input_acceleration
   + 0.9730004072189331 x2_input_acceleration
   - 0.00600028038024902 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.328000009059906 x1_input_acceleration
   + 0.7000001430511471 x2_input_acceleration
   + 0.1179993867874145 x3_input_acceleration >= -2.804877028964142
 Relaxed_CLF_constraint: - 0.6959998309612274 x1_input_acceleration
   + 0.9530004262924194 x2_input_acceleration
   + 0.0139997005462646 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.287999987602234 x1_input_acceleration
   + 0.6600001811981198 x2_input_acceleration
   + 0.0779994249343872 x3_input_acceleration >= -2.723157016985984
 Relaxed_CLF_constraint: - 0.6759998202323914 x1_input_acceleration
   + 0.9330004453659058 x2_input_acceleration
   - 0.00600028038024902 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.247999966144562 x1_input_acceleration
   + 0.6200002193450924 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -2.646956969803003
 Relaxed_CLF_constraint: - 0.6559998095035553 x1_input_acceleration
   + 0.9130004644393921 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.20799994468689 x1_input_acceleration
   + 0.580000019073486 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -2.568436889732482
 Relaxed_CLF_constraint: - 0.6359997987747192 x1_input_acceleration
   + 0.8930003643035889 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.167999923229218 x1_input_acceleration
   + 0.5400000572204586 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -2.49543685063185
 Relaxed_CLF_constraint: - 0.6159997880458832 x1_input_acceleration
   + 0.8730003833770752 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.127999901771545 x1_input_acceleration
   + 0.5000000953674313 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -2.420116843135959
 Relaxed_CLF_constraint: - 0.5959997773170471 x1_input_acceleration
   + 0.8530004024505615 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.087999880313873 x1_input_acceleration
   + 0.4600001335144039 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -2.350316802700185
 Relaxed_CLF_constraint: - 0.5759997665882111 x1_input_acceleration
   + 0.8330004215240479 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.047999858856201 x1_input_acceleration
   + 0.4200001716613766 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -2.278196793869152
 Relaxed_CLF_constraint: - 0.555999755859375 x1_input_acceleration
   + 0.8130004405975342 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 3.007999837398529 x1_input_acceleration
   + 0.3800002098083493 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -2.211596752098235
 Relaxed_CLF_constraint: - 0.5359997451305389 x1_input_acceleration
   + 0.7930004596710205 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.967999815940857 x1_input_acceleration
   + 0.3400002479553219 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -2.14267674193206
 Relaxed_CLF_constraint: - 0.5159997344017029 x1_input_acceleration
   + 0.7730004787445068 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.92799985408783 x1_input_acceleration
   + 0.3000002861022946 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -2.079276786087196
 Relaxed_CLF_constraint: - 0.4959997534751892 x1_input_acceleration
   + 0.7530004978179932 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.887999892234802 x1_input_acceleration
   + 0.2600003242492672 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -2.013556859462888
 Relaxed_CLF_constraint: - 0.4759997725486755 x1_input_acceleration
   + 0.7330005168914795 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.847999930381775 x1_input_acceleration
   + 0.2200003623962399 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -1.953356897514511
 Relaxed_CLF_constraint: - 0.4559997916221619 x1_input_acceleration
   + 0.7130005359649658 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.807999968528748 x1_input_acceleration
   + 0.1800004005432125 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -1.89083696478669
 Relaxed_CLF_constraint: - 0.4359998106956482 x1_input_acceleration
   + 0.6930005550384521 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.76800000667572 x1_input_acceleration
   + 0.1400004386901852 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -1.833836996734801
 Relaxed_CLF_constraint: - 0.4159998297691345 x1_input_acceleration
   + 0.6730005741119385 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.728000044822693 x1_input_acceleration
   + 0.1000004768371578 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -1.774517057903468
 Relaxed_CLF_constraint: - 0.3959998488426208 x1_input_acceleration
   + 0.6530005931854248 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.688000082969666 x1_input_acceleration
   + 0.0600005149841305 x2_input_acceleration
   + 0.117999267578125 x3_input_acceleration >= -1.720717083748066
 Relaxed_CLF_constraint: - 0.3759998679161072 x1_input_acceleration
   + 0.6330006122589111 x2_input_acceleration
   + 0.0139996409416199 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.648000121116638 x1_input_acceleration
   + 0.0200005531311032 x2_input_acceleration
   + 0.0779993057250976 x3_input_acceleration >= -1.664597138813219
 Relaxed_CLF_constraint: - 0.3559998869895935 x1_input_acceleration
   + 0.6130006313323975 x2_input_acceleration
   - 0.00600034 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.608000159263611 x1_input_acceleration
   - 0.0199994087219242 x2_input_acceleration
   + 0.1179991483688354 x3_input_acceleration >= -1.613997151521004
 Relaxed_CLF_constraint: - 0.3359999060630798 x1_input_acceleration
   + 0.5930006504058838 x2_input_acceleration
   + 0.0139995813369751 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.568000197410583 x1_input_acceleration
   - 0.0599993705749515 x2_input_acceleration
   + 0.0779991865158081 x3_input_acceleration >= -1.561077202866828
 Relaxed_CLF_constraint: - 0.3159999251365662 x1_input_acceleration
   + 0.5730006694793701 x2_input_acceleration
   - 0.00600039958953857 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.528000235557556 x1_input_acceleration
   - 0.0999993324279789 x2_input_acceleration
   + 0.1179991483688354 x3_input_acceleration >= -1.513677214120218
 Relaxed_CLF_constraint: - 0.2959999442100525 x1_input_acceleration
   + 0.5530006885528564 x2_input_acceleration
   + 0.0139995813369751 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.488000273704529 x1_input_acceleration
   - 0.1399992942810062 x2_input_acceleration
   + 0.0779991865158081 x3_input_acceleration >= -1.463957259362529
 Relaxed_CLF_constraint: - 0.2759999632835388 x1_input_acceleration
   + 0.5330007076263428 x2_input_acceleration
   - 0.00600039958953857 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.448000311851501 x1_input_acceleration
   - 0.1799992561340336 x2_input_acceleration
   + 0.1179990291595459 x3_input_acceleration >= -1.419757257479112
 Relaxed_CLF_constraint: - 0.2559999823570251 x1_input_acceleration
   + 0.5130007266998291 x2_input_acceleration
   + 0.0139995217323303 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.408000349998474 x1_input_acceleration
   - 0.2199992179870609 x2_input_acceleration
   + 0.0779990673065185 x3_input_acceleration >= -1.373237299002095
 Relaxed_CLF_constraint: - 0.2360000014305115 x1_input_acceleration
   + 0.4930007457733154 x2_input_acceleration
   - 0.00600045919418335 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.368000388145447 x1_input_acceleration
   - 0.2599991798400882 x2_input_acceleration
   + 0.1179990291595459 x3_input_acceleration >= -1.332237295664275
 Relaxed_CLF_constraint: - 0.2160000205039978 x1_input_acceleration
   + 0.4730007648468018 x2_input_acceleration
   + 0.0139995217323303 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.328000426292419 x1_input_acceleration
   - 0.2999991416931156 x2_input_acceleration
   + 0.0779990673065185 x3_input_acceleration >= -1.288917331083745
 Relaxed_CLF_constraint: - 0.1960000395774841 x1_input_acceleration
   + 0.4530007839202881 x2_input_acceleration
   - 0.00600045919418335 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.288000464439392 x1_input_acceleration
   - 0.3399991035461429 x2_input_acceleration
   + 0.1179990291595459 x3_input_acceleration >= -1.251117321642413
 Relaxed_CLF_constraint: - 0.1760000586509705 x1_input_acceleration
   + 0.4330008029937744 x2_input_acceleration
   + 0.0139995217323303 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.248000502586365 x1_input_acceleration
   - 0.3799990653991703 x2_input_acceleration
   + 0.0779990673065185 x3_input_acceleration >= -1.210997350958369
 Relaxed_CLF_constraint: - 0.1560000777244568 x1_input_acceleration
   + 0.4130008220672607 x2_input_acceleration
   - 0.00600045919418335 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.208000540733337 x1_input_acceleration
   - 0.4199990272521976 x2_input_acceleration
   + 0.1179978370666503 x3_input_acceleration >= -1.176397265080978
 Relaxed_CLF_constraint: - 0.1360000967979431 x1_input_acceleration
   + 0.3930008411407471 x2_input_acceleration
   + 0.0139989256858826 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.16800057888031 x1_input_acceleration
   - 0.459998989105225 x2_input_acceleration
   + 0.077997875213623 x3_input_acceleration >= -1.139477312135257
 Relaxed_CLF_constraint: - 0.1160001158714294 x1_input_acceleration
   + 0.3730008602142334 x2_input_acceleration
   - 0.0060010552406311 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.128000617027283 x1_input_acceleration
   - 0.4999989509582523 x2_input_acceleration
   + 0.1179978370666503 x3_input_acceleration >= -1.108077266645064
 Relaxed_CLF_constraint: - 0.0960001349449158 x1_input_acceleration
   + 0.3530008792877197 x2_input_acceleration
   + 0.0139989256858826 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.088000655174255 x1_input_acceleration
   - 0.5399989128112797 x2_input_acceleration
   + 0.077997875213623 x3_input_acceleration >= -1.074357307595831
 Relaxed_CLF_constraint: - 0.0760001540184021 x1_input_acceleration
   + 0.3330008983612061 x2_input_acceleration
   - 0.0060010552406311 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.048000693321228 x1_input_acceleration
   - 0.579998874664307 x2_input_acceleration
   + 0.1179978370666503 x3_input_acceleration >= -1.046157256002125
 Relaxed_CLF_constraint: - 0.0560001730918884 x1_input_acceleration
   + 0.3130009174346924 x2_input_acceleration
   + 0.0139989256858826 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 2.008000731468201 x1_input_acceleration
   - 0.6199988365173343 x2_input_acceleration
   + 0.077997875213623 x3_input_acceleration >= -1.015637290849379
 Relaxed_CLF_constraint: - 0.0360001921653748 x1_input_acceleration
   + 0.2930009365081787 x2_input_acceleration
   - 0.0060010552406311 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.968000650405884 x1_input_acceleration
   - 0.6599987983703617 x2_input_acceleration
   + 0.1179964065551757 x3_input_acceleration >= -0.9906370314520595
 Relaxed_CLF_constraint: - 0.0160001516342163 x1_input_acceleration
   + 0.273000955581665 x2_input_acceleration
   + 0.0139982104301453 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.928000688552856 x1_input_acceleration
   - 0.699998760223389 x2_input_acceleration
   + 0.0779964447021484 x3_input_acceleration >= -0.9633170911901864
 Relaxed_CLF_constraint: 0.00399982929229736 x1_input_acceleration
   + 0.2530009746551514 x2_input_acceleration
   - 0.00600177049636841 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.939682483673096 x1_input_acceleration
   - 0.7399987220764164 x2_input_acceleration
   + 0.1063144922256469 x3_input_acceleration >= -0.990317254350039
 Relaxed_CLF_constraint: - 0.00184106826782227 x1_input_acceleration
   + 0.2330009937286377 x2_input_acceleration
   + 0.00815725326538086 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.924381971359253 x1_input_acceleration
   - 0.7799986839294437 x2_input_acceleration
   + 0.0663145303726196 x3_input_acceleration >= -0.9890103838906824
 Relaxed_CLF_constraint: 0.00580918788909912 x1_input_acceleration
   + 0.213001012802124 x2_input_acceleration
   - 0.0118427276611328 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.947003602981567 x1_input_acceleration
   - 0.8199986457824711 x2_input_acceleration
   + 0.0976614475250244 x3_input_acceleration >= -1.028189641860243
 Relaxed_CLF_constraint: - 0.00550162792205811 x1_input_acceleration
   + 0.1930010318756104 x2_input_acceleration
   + 0.00383073091506958 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.907822847366333 x1_input_acceleration
   - 0.8599986076354984 x2_input_acceleration
   + 0.0748947381973266 x3_input_acceleration >= -1.006248710969406
 Relaxed_CLF_constraint: 0.0140887498855591 x1_input_acceleration
   + 0.1730010509490967 x2_input_acceleration
   - 0.0075526237487793 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.940640211105347 x1_input_acceleration
   - 0.8999985694885257 x2_input_acceleration
   + 0.0886952400207519 x3_input_acceleration >= -1.055987174910684
 Relaxed_CLF_constraint: - 0.00231993198394775 x1_input_acceleration
   + 0.153001070022583 x2_input_acceleration
   - 6.5237283706665039e-04 x3_input_acceleration + relaxation_CLF
   <= -20.8
 CBF_Constraint: - 1.921352624893188 x1_input_acceleration
   - 0.9399985313415531 x2_input_acceleration
   + 0.0941189050674438 x3_input_acceleration >= -1.056012879099829
 Relaxed_CLF_constraint: 0.00732386112213135 x1_input_acceleration
   + 0.1330010890960693 x2_input_acceleration
   + 0.0020594596862793 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.961352586746216 x1_input_acceleration
   - 0.9799984931945804 x2_input_acceleration
   + 0.0769958019256591 x3_input_acceleration >= -1.113307342428474
 Relaxed_CLF_constraint: - 0.0126761198043823 x1_input_acceleration
   + 0.1130011081695557 x2_input_acceleration
   - 0.00650209188461304 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.93078351020813 x1_input_acceleration
   - 1.019998455047608 x2_input_acceleration
   + 0.0990263938903808 x3_input_acceleration >= -1.104532009569517
 Relaxed_CLF_constraint: 0.00260841846466064 x1_input_acceleration
   + 0.093001127243042 x2_input_acceleration
   + 0.0045132040977478 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.95246958732605 x1_input_acceleration
   - 1.059998416900635 x2_input_acceleration
   + 0.0615058660507202 x3_input_acceleration >= -1.144879276205914
 Relaxed_CLF_constraint: - 0.00823462 x1_input_acceleration
   + 0.0730011463165283 x2_input_acceleration
   - 0.0142470598220825 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.933993816375732 x1_input_acceleration
   - 1.099998378753662 x2_input_acceleration
   + 0.0934705257415771 x3_input_acceleration >= -1.149766313555666
 Relaxed_CLF_constraint: 0.00100326538085938 x1_input_acceleration
   + 0.0530011653900146 x2_input_acceleration
   + 0.00173527 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.942338705062866 x1_input_acceleration
   - 1.13999834060669 x2_input_acceleration
   + 0.0790367841720581 x3_input_acceleration >= -1.17963066875589
 Relaxed_CLF_constraint: - 0.00316917896270752 x1_input_acceleration
   + 0.033001184463501 x2_input_acceleration
   - 0.00548160076141357 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.915976285934448 x1_input_acceleration
   - 1.179998302459717 x2_input_acceleration
   + 0.1190367460250854 x3_input_acceleration >= -1.179382717243804
 Relaxed_CLF_constraint: 0.0100120306015015 x1_input_acceleration
   + 0.0130012035369873 x2_input_acceleration
   + 0.0145183801651001 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.936927795410156 x1_input_acceleration
   - 1.219998264312744 x2_input_acceleration
   + 0.0928450345993042 x3_input_acceleration >= -1.222176312502076
 Relaxed_CLF_constraint: - 4.6372413635253906e-04 x1_input_acceleration
   - 0.00699877738952637 x2_input_acceleration
   + 0.00142252445220947 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.932918787002563 x1_input_acceleration
   - 1.220564270019532 x2_input_acceleration
   + 0.0879952430725097 x3_input_acceleration >= -1.218423834299791
 Relaxed_CLF_constraint: 0.00154078 x1_input_acceleration
   - 0.00728178024291992 x2_input_acceleration
   - 0.00100237131118774 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.932918787002563 x1_input_acceleration
   - 1.220564270019532 x2_input_acceleration
   + 0.0879952430725097 x3_input_acceleration >= -1.218423834299791
 Relaxed_CLF_constraint: 0.00154078 x1_input_acceleration
   - 0.00728178024291992 x2_input_acceleration
   - 0.00100237131118774 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.932918787002563 x1_input_acceleration
   - 1.220564270019532 x2_input_acceleration
   + 0.0879952430725097 x3_input_acceleration >= -1.218423834299791
 Relaxed_CLF_constraint: 0.00154078 x1_input_acceleration
   - 0.00728178024291992 x2_input_acceleration
   - 0.00100237131118774 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.932918787002563 x1_input_acceleration
   - 1.220564270019532 x2_input_acceleration
   + 0.0879952430725097 x3_input_acceleration >= -1.218423834299791
 Relaxed_CLF_constraint: 0.00154078 x1_input_acceleration
   - 0.00728178024291992 x2_input_acceleration
   - 0.00100237131118774 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.932918787002563 x1_input_acceleration
   - 1.220564270019532 x2_input_acceleration
   + 0.0879952430725097 x3_input_acceleration >= -1.218423834299791
 Relaxed_CLF_constraint: 0.00154078 x1_input_acceleration
   - 0.00728178024291992 x2_input_acceleration
   - 0.00100237131118774 x3_input_acceleration + relaxation_CLF <= -20.8
 CBF_Constraint: - 1.932918787002563 x1_input_acceleration
   - 1.220564270019532 x2_input_acceleration
   + 0.0879952430725097 x3_input_acceleration >= -1.218423834299791
 Relaxed_CLF_constraint: 0.00154078 x1_input_acceleration
   - 0.00728178024291992 x2_input_acceleration
   - 0.00100237131118774 x3_input_acceleration + relaxation_CLF <= -20.8
Bounds
 -0.1 <= x1_input_acceleration <= 0.1
 -0.1 <= x2_input_acceleration <= 0.1
 -0.1 <= x3_input_acceleration <= 0.1
 -30 <= relaxation_CLF <= 30
End

\ Model CBF_CLF_QP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration
   + 0 x1_input_acceleration + 0 x1_input_acceleration + [
   2 x1_input_acceleration ^2 + 2 x2_input_acceleration ^2
   + 2 x3_input_acceleration ^2 + 2 x1_input_acceleration ^2 ] / 2 
Subject To
 Relaxed_CLF_constraint: - 0.5 x1_input_acceleration
   - 0.5 x2_input_acceleration <= -0.025
 Relaxed_CLF_constraint: - 0.4824999868869781 x1_input_acceleration
   - 0.4825000762939453 x2_input_acceleration <= -0.0232806280484799
 Relaxed_CLF_constraint: - 0.4649999737739563 x1_input_acceleration
   - 0.4650001525878906 x2_input_acceleration <= -0.0216225058758271
 Relaxed_CLF_constraint: - 0.4474999606609344 x1_input_acceleration
   - 0.4475002288818359 x2_input_acceleration <= -0.0200256334820417
 Relaxed_CLF_constraint: - 0.4299999475479126 x1_input_acceleration
   - 0.4300003051757812 x2_input_acceleration <= -0.0184900108671236
 Relaxed_CLF_constraint: - 0.4124999642372131 x1_input_acceleration
   - 0.412500262260437 x2_input_acceleration <= -0.0170156343430316
 Relaxed_CLF_constraint: - 0.3949999809265137 x1_input_acceleration
   - 0.3950002193450928 x2_input_acceleration <= -0.0156025079107309
 Relaxed_CLF_constraint: - 0.3774999976158142 x1_input_acceleration
   - 0.3775001764297485 x2_input_acceleration <= -0.0142506315702216
 Relaxed_CLF_constraint: - 0.3600000143051147 x1_input_acceleration
   - 0.3600001335144043 x2_input_acceleration <= -0.0129600053215036
 Relaxed_CLF_constraint: - 0.3425000309944153 x1_input_acceleration
   - 0.3425000905990601 x2_input_acceleration <= -0.011730629164577
 Relaxed_CLF_constraint: - 0.3250000476837158 x1_input_acceleration
   - 0.3250000476837158 x2_input_acceleration <= -0.0105625030994418
 Relaxed_CLF_constraint: - 0.3075000643730164 x1_input_acceleration
   - 0.3075000047683716 x2_input_acceleration <= -0.00945562712609789
 Relaxed_CLF_constraint: - 0.2900000810623169 x1_input_acceleration
   - 0.2899999618530273 x2_input_acceleration <= -0.00841000124454538
 Relaxed_CLF_constraint: - 0.2725000977516174 x1_input_acceleration
   - 0.2724999189376831 x2_input_acceleration <= -0.00742562545478425
 Relaxed_CLF_constraint: - 0.255000114440918 x1_input_acceleration
   - 0.2549998760223389 x2_input_acceleration <= -0.00650249975681447
 Relaxed_CLF_constraint: - 0.2375001311302185 x1_input_acceleration
   - 0.2374998331069946 x2_input_acceleration <= -0.00564062415063606
 Relaxed_CLF_constraint: - 0.220000147819519 x1_input_acceleration
   - 0.2199997901916504 x2_input_acceleration <= -0.00483999863624902
 Relaxed_CLF_constraint: - 0.2025001645088196 x1_input_acceleration
   - 0.2024997472763062 x2_input_acceleration <= -0.00410062321365334
 Relaxed_CLF_constraint: - 0.1850001811981201 x1_input_acceleration
   - 0.1849997043609619 x2_input_acceleration <= -0.00342249788284903
 Relaxed_CLF_constraint: - 0.1675001978874207 x1_input_acceleration
   - 0.1674996614456177 x2_input_acceleration <= -0.00280562264383608
 Relaxed_CLF_constraint: - 0.1500002145767212 x1_input_acceleration
   - 0.1499996185302734 x2_input_acceleration <= -0.0022499974966145
 Relaxed_CLF_constraint: - 0.1325002312660217 x1_input_acceleration
   - 0.1324995756149292 x2_input_acceleration <= -0.00175562244118428
 Relaxed_CLF_constraint: - 0.1150002479553223 x1_input_acceleration
   - 0.114999532699585 x2_input_acceleration <= -0.00132249747754543
 Relaxed_CLF_constraint: - 0.0975002646446228 x1_input_acceleration
   - 0.0974994897842407 x2_input_acceleration <= -9.5062260569793726e-04
 Relaxed_CLF_constraint: - 0.0800002813339233 x1_input_acceleration
   - 0.0799994468688965 x2_input_acceleration <= -6.3999782564181382e-04
 Relaxed_CLF_constraint: - 0.0625002980232239 x1_input_acceleration
   - 0.0624994039535522 x2_input_acceleration <= -3.9062313737705525e-04
 Relaxed_CLF_constraint: - 0.0450003147125244 x1_input_acceleration
   - 0.044999361038208 x2_input_acceleration <= -2.0249854090366172e-04
 Relaxed_CLF_constraint: - 0.027500331401825 x1_input_acceleration
   - 0.0274993181228638 x2_input_acceleration <= -7.5624e-05
 Relaxed_CLF_constraint: - 0.0100003480911255 x1_input_acceleration
   - 0.00999927520751953 x2_input_acceleration <= -9.99962e-06
Bounds
 -0.1 <= x1_input_acceleration <= 0.1
 -0.1 <= x2_input_acceleration <= 0.1
 -0.1 <= x3_input_acceleration <= 0.1
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
 -5 <= x1_input_acceleration <= 5
End

# utility file to convert csv file into nice table format
import pandas as pd


def fill_block_exp2_eb(list, list_e):
    prec = 3
    prec_e = 3
    table_string = (
        f"& Linear     & {list[0]:.{prec}f}{{\\tiny$\\pm${list_e[0]:.{prec_e}f}}} & {list[1]:.{prec}f}{{\\tiny$\\pm${list_e[1]:.{prec_e}f}}} & {list[2]:.{prec}f}{{\\tiny$\\pm${list_e[2]:.{prec_e}f}}} \\\\\n"
        f"& & Polynomial & {list[3]:.{prec}f}{{\\tiny$\\pm${list_e[3]:.{prec_e}f}}} & {list[4]:.{prec}f}{{\\tiny$\\pm${list_e[4]:.{prec_e}f}}} & {list[5]:.{prec}f}{{\\tiny$\\pm${list_e[5]:.{prec_e}f}}} \\\\\n"
        f"& & Nonlinear  & {list[6]:.{prec}f}{{\\tiny$\\pm${list_e[6]:.{prec_e}f}}} & {list[7]:.{prec}f}{{\\tiny$\\pm${list_e[7]:.{prec_e}f}}} & {list[8]:.{prec}f}{{\\tiny$\\pm${list_e[8]:.{prec_e}f}}} \\\\\n"
    )
    return table_string

def fill_block_exp2(list):
    table_string = (
        f"& Linear     & {list[0]:.5f} & {list[1]:.5f} & {list[2]:.5f} \\\\\n"
        f"& & Polynomial & {list[3]:.5f} & {list[4]:.5f} & {list[5]:.5f} \\\\\n"
        f"& & Nonlinear  & {list[6]:.5f} & {list[7]:.5f} & {list[8]:.5f} \\\\\n"
    )
    return table_string

def fill_block_exp2_long(list, list2, list3):
    table_string = (
        f"& Linear       & {list[0]:.5f} & {list2[0]:.5f} & {list3[0]:.5f} & {list[1]:.5f} & {list2[1]:.5f} & {list3[1]:.5f} & {list[2]:.5f} & {list2[2]:.5f} & {list3[2]:.5f} \\\\\n"
        f"& & Polynomial & {list[3]:.5f} & {list2[3]:.5f} & {list3[3]:.5f} & {list[4]:.5f} & {list2[4]:.5f} & {list3[4]:.5f} & {list[5]:.5f} & {list2[5]:.5f} & {list3[5]:.5f} \\\\\n"
        f"& & Nonlinear  & {list[6]:.5f} & {list2[6]:.5f} & {list3[6]:.5f} & {list[7]:.5f} & {list2[7]:.5f} & {list3[7]:.5f} & {list[8]:.5f} & {list2[8]:.5f} & {list3[8]:.5f} \\\\\n"
    )
    return table_string

def fill_block_exp2_long_eb(list, list2, list3, list_e, list2_e, list3_e):
    prec = 3
    prec_e = 3
    table_string = (
        f"& Linear       & {list[0]:.{prec}f}{{\\tiny$\\pm${list_e[0]:.{prec_e}f}}} & {list2[0]:.{prec}f}{{\\tiny$\\pm${list2_e[0]:.{prec_e}f}}} & {list3[0]:.{prec}f}{{\\tiny$\\pm${list3_e[0]:.{prec_e}f}}} & {list[1]:.{prec}f}{{\\tiny$\\pm${list_e[1]:.{prec_e}f}}} & {list2[1]:.{prec}f}{{\\tiny$\\pm${list2_e[1]:.{prec_e}f}}} & {list3[1]:.{prec}f}{{\\tiny$\\pm${list3_e[1]:.{prec_e}f}}} & {list[2]:.{prec}f}{{\\tiny$\\pm${list_e[2]:.{prec_e}f}}} & {list2[2]:.{prec}f}{{\\tiny$\\pm${list2_e[2]:.{prec_e}f}}} & {list3[2]:.{prec}f}{{\\tiny$\\pm${list3_e[2]:.{prec_e}f}}} \\\\\n"
        f"& & Polynomial & {list[3]:.{prec}f}{{\\tiny$\\pm${list_e[3]:.{prec_e}f}}} & {list2[3]:.{prec}f}{{\\tiny$\\pm${list2_e[3]:.{prec_e}f}}} & {list3[3]:.{prec}f}{{\\tiny$\\pm${list3_e[3]:.{prec_e}f}}} & {list[4]:.{prec}f}{{\\tiny$\\pm${list_e[4]:.{prec_e}f}}} & {list2[4]:.{prec}f}{{\\tiny$\\pm${list2_e[4]:.{prec_e}f}}} & {list3[4]:.{prec}f}{{\\tiny$\\pm${list3_e[4]:.{prec_e}f}}} & {list[5]:.{prec}f}{{\\tiny$\\pm${list_e[5]:.{prec_e}f}}} & {list2[5]:.{prec}f}{{\\tiny$\\pm${list2_e[5]:.{prec_e}f}}} & {list3[5]:.{prec}f}{{\\tiny$\\pm${list3_e[5]:.{prec_e}f}}} \\\\\n"
        f"& & Nonlinear & {list[6]:.{prec}f}{{\\tiny$\\pm${list_e[6]:.{prec_e}f}}} & {list2[6]:.{prec}f}{{\\tiny$\\pm${list2_e[6]:.{prec_e}f}}} & {list3[6]:.{prec}f}{{\\tiny$\\pm${list3_e[6]:.{prec_e}f}}} & {list[7]:.{prec}f}{{\\tiny$\\pm${list_e[7]:.{prec_e}f}}} & {list2[7]:.{prec}f}{{\\tiny$\\pm${list2_e[7]:.{prec_e}f}}} & {list3[7]:.{prec}f}{{\\tiny$\\pm${list3_e[7]:.{prec_e}f}}} & {list[8]:.{prec}f}{{\\tiny$\\pm${list_e[8]:.{prec_e}f}}} & {list2[8]:.{prec}f}{{\\tiny$\\pm${list2_e[8]:.{prec_e}f}}} & {list3[8]:.{prec}f}{{\\tiny$\\pm${list3_e[8]:.{prec_e}f}}} \\\\\n"
    )
    return table_string


def convert_exp2_eb(filename):
    df = pd.read_csv(filename)
    cf = df['mse_closed_form_mean']
    l_cf = [cf[0], cf[24], cf[12], cf[2], cf[26], cf[14], cf[1], cf[25], cf[13]]
    cf = df['mse_closed_form_ci95']
    l_cf_e = [cf[0], cf[24], cf[12], cf[2], cf[26], cf[14], cf[1], cf[25], cf[13]]

    cf_block = fill_block_exp2_eb(l_cf, l_cf_e)

    col = df['mse_gradient_descent_mean']
    col_e = df['mse_gradient_descent_ci95']
    blocks=[]
    for i in range(4):
        l = [col[0+i*3], col[24+i*3], col[12+i*3], col[2+i*3], col[26+i*3], col[14+i*3], col[1+i*3], col[25+i*3], col[13+i*3]]
        l_e = [col_e[0+i*3], col_e[24+i*3], col_e[12+i*3], col_e[2+i*3], col_e[26+i*3], col_e[14+i*3], col_e[1+i*3], col_e[25+i*3], col_e[13+i*3]]

        block = fill_block_exp2_eb(l, l_e)
        blocks.append(block)

    [rev_kl_block, fwd_kl_block, mle_ds_block, mle_pa_block] = blocks

    out = rf"""\begin{{table*}}[h!]
    \centering
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.2}}
    \begin{{tabular}}{{lllccc}}
    \toprule
    \textbf{{Model Family}} & \textbf{{Objective}} & \textbf{{Modelling Assumption}} & \multicolumn{{3}}{{c}}{{Loss($\downarrow$)}}\\
    \cmidrule(lr){{4-6}}
    & & & Linear & Polynomial & Nonlinear \\
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \multirow{{3}}{{*}}{{Closed Form Solution}} 
    & \multirow{{3}}{{*}}{{MLE}} 
    {cf_block}
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \multirow{{12}}{{*}}{{Amortized}} 
    & \multirow{{3}}{{*}}{{Rev-KL}} 
    {rev_kl_block}
    \cmidrule{{2-6}}
    & \multirow{{3}}{{*}}{{Fwd-KL}} 
    {fwd_kl_block}
    \cmidrule{{2-6}}
    & \multirow{{3}}{{*}}{{MLE-Dataset}}
    {mle_ds_block}
    \cmidrule{{2-6}}
    & \multirow{{3}}{{*}}{{MLE-Params}}
    {mle_pa_block}
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \end{{tabular}}
    \caption{{}}
    \end{{table*}}"""

    print(out)

def convert_exp2(filename):
    df = pd.read_csv(filename)
    cf = df['mse_closed_form']
    l_cf = [cf[0], cf[24], cf[12], cf[2], cf[26], cf[14], cf[1], cf[25], cf[13]]
    cf_block = fill_block_exp2(l_cf)

    col = df['mse_gradient_descent']
    blocks=[]
    for i in range(4):
        l = [col[0+i*3], col[24+i*3], col[12+i*3], col[2+i*3], col[26+i*3], col[14+i*3], col[1+i*3], col[25+i*3], col[13+i*3]]
        block = fill_block_exp2(l)
        blocks.append(block)

    [rev_kl_block, fwd_kl_block, mle_ds_block, mle_pa_block] = blocks

    out = rf"""\begin{{table*}}[h!]
    \centering
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.2}}
    \begin{{tabular}}{{lllccc}}
    \toprule
    \textbf{{Model Family}} & \textbf{{Objective}} & \textbf{{Modelling Assumption}} & \multicolumn{{3}}{{c}}{{Loss($\downarrow$)}}\\
    \cmidrule(lr){{4-6}}
    & & & Linear & Polynomial & Nonlinear \\
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \multirow{{3}}{{*}}{{Closed Form Solution}} 
    & \multirow{{3}}{{*}}{{MLE}} 
    {cf_block}
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \multirow{{12}}{{*}}{{Amortized}} 
    & \multirow{{3}}{{*}}{{Rev-KL}} 
    {rev_kl_block}
    \cmidrule{{2-6}}
    & \multirow{{3}}{{*}}{{Fwd-KL}} 
    {fwd_kl_block}
    \cmidrule{{2-6}}
    & \multirow{{3}}{{*}}{{MLE-Dataset}}
    {mle_ds_block}
    \cmidrule{{2-6}}
    & \multirow{{3}}{{*}}{{MLE-Params}}
    {mle_pa_block}
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \end{{tabular}}
    \caption{{}}
    \end{{table*}}"""

    print(out)


def convert_exp2_ablation_eb(filename, filename_inc_ds, filename_inc_params):
    df = pd.read_csv(filename)
    df_ds = pd.read_csv(filename_inc_ds)
    df_params = pd.read_csv(filename_inc_params)
    cf = df['mse_closed_form_mean']
    cf_ds = df_ds['mse_closed_form_mean']
    cf_params = df_params['mse_closed_form_mean']
    l_cf = [cf[0], cf[24], cf[12], cf[2], cf[26], cf[14], cf[1], cf[25], cf[13]]
    l_cf_ds = [cf_ds[0], cf_ds[24], cf_ds[12], cf_ds[2], cf_ds[26], cf_ds[14], cf_ds[1], cf_ds[25], cf_ds[13]]
    l_cf_params = [cf_params[0], cf_params[24], cf_params[12], cf_params[2], cf_params[26], cf_params[14], cf_params[1], cf_params[25], cf_params[13]]
    cf = df['mse_closed_form_ci95']
    cf_ds = df_ds['mse_closed_form_ci95']
    cf_params = df_params['mse_closed_form_ci95']
    l_cf_e = [cf[0], cf[24], cf[12], cf[2], cf[26], cf[14], cf[1], cf[25], cf[13]]
    l_cf_ds_e = [cf_ds[0], cf_ds[24], cf_ds[12], cf_ds[2], cf_ds[26], cf_ds[14], cf_ds[1], cf_ds[25], cf_ds[13]]
    l_cf_params_e = [cf_params[0], cf_params[24], cf_params[12], cf_params[2], cf_params[26], cf_params[14], cf_params[1],
                   cf_params[25], cf_params[13]]

    cf_block = fill_block_exp2_long_eb(l_cf, l_cf_params, l_cf_ds, l_cf_e, l_cf_params_e, l_cf_ds_e)

    col = df['mse_gradient_descent_mean']
    col_ds = df_ds['mse_gradient_descent_mean']
    col_params = df_params['mse_gradient_descent_mean']
    col_e = df['mse_gradient_descent_ci95']
    col_ds_e = df_ds['mse_gradient_descent_ci95']
    col_params_e = df_params['mse_gradient_descent_ci95']
    blocks = []
    for i in range(4):
        l = [col[0 + i * 3], col[24 + i * 3], col[12 + i * 3], col[2 + i * 3], col[26 + i * 3], col[14 + i * 3],
             col[1 + i * 3], col[25 + i * 3], col[13 + i * 3]]
        l_ds = [col_ds[0 + i * 3], col_ds[24 + i * 3], col_ds[12 + i * 3], col_ds[2 + i * 3], col_ds[26 + i * 3], col_ds[14 + i * 3],
             col_ds[1 + i * 3], col_ds[25 + i * 3], col_ds[13 + i * 3]]
        l_params = [col_params[0 + i * 3], col_params[24 + i * 3], col_params[12 + i * 3], col_params[2 + i * 3], col_params[26 + i * 3], col_params[14 + i * 3],
             col_params[1 + i * 3], col_params[25 + i * 3], col_params[13 + i * 3]]

        l_e = [col_e[0 + i * 3], col_e[24 + i * 3], col_e[12 + i * 3], col_e[2 + i * 3], col_e[26 + i * 3], col_e[14 + i * 3],
             col_e[1 + i * 3], col_e[25 + i * 3], col_e[13 + i * 3]]
        l_ds_e = [col_ds_e[0 + i * 3], col_ds_e[24 + i * 3], col_ds_e[12 + i * 3], col_ds_e[2 + i * 3], col_ds_e[26 + i * 3],
                col_ds_e[14 + i * 3],
                col_ds_e[1 + i * 3], col_ds_e[25 + i * 3], col_ds_e[13 + i * 3]]
        l_params_e = [col_params_e[0 + i * 3], col_params_e[24 + i * 3], col_params_e[12 + i * 3], col_params_e[2 + i * 3],
                    col_params_e[26 + i * 3], col_params_e[14 + i * 3],
                    col_params_e[1 + i * 3], col_params_e[25 + i * 3], col_params_e[13 + i * 3]]

        block = fill_block_exp2_long_eb(l, l_params, l_ds, l_e, l_params_e, l_ds_e)
        blocks.append(block)
    [rev_kl_block, fwd_kl_block, mle_ds_block, mle_pa_block] = blocks

    out = rf"""\begin{{table*}}[h!]
    \centering
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.2}}
    \begin{{tabular}}{{lllccccccccc}}
    \toprule
    \textbf{{Model Family}} & \textbf {{Objective}} & \textbf {{Modelling Assumption}} & \multicolumn {{9}}{{c}}{{$L_2$ Loss($\downarrow$)}} \\
    \cmidrule(lr){{4 - 12}}
    & & & \multicolumn{{3}}{{c}}{{Linear}} & \multicolumn {{3}}{{c}}{{Polynomial}} & \multicolumn{{3}}{{c}}{{Nonlinear}} \\
    \cmidrule(lr){{4 - 6}}\cmidrule(lr){{7 - 9}}\cmidrule(lr){{10 - 12}} & & & Default & Inc.Parameters & Inc.Dataset Size & Default & Inc.Parameters & Inc.Dataset Size & Default & Inc.Parameters & Inc.Dataset Size \\
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \multirow{{3}}{{*}}{{Closed Form Solution}} & \multirow{{3}}{{*}}{{MLE}}\
    {cf_block}
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}\multirow{{12}}{{*}}{{Amortized}} & \multirow{{3}}{{*}}{{Rev-KL}}
    {rev_kl_block}
    \cmidrule{{2 - 12}}
    &  \multirow{{3}}{{*}}{{Fwd-KL}}
    {fwd_kl_block}
    \cmidrule{{2 - 12}}
    &  \multirow{{3}}{{*}}{{MLE-Dataset}}
    {mle_ds_block}
    \cmidrule{{2 - 12}}
    &  \multirow{{3}}{{*}}{{MLE-Params}}
    {mle_pa_block}
    \end{{tabular}}\caption{{}}\end{{table*}}
    """
    print(out)

def convert_exp2_ablation(filename, filename_inc_ds, filename_inc_params):
    df = pd.read_csv(filename)
    df_ds = pd.read_csv(filename_inc_ds)
    df_params = pd.read_csv(filename_inc_params)
    df_params[:] = float('nan')
    cf = df['mse_closed_form']
    cf_ds = df_ds['mse_closed_form']
    cf_params = df_params['mse_closed_form']
    l_cf = [cf[0], cf[24], cf[12], cf[2], cf[26], cf[14], cf[1], cf[25], cf[13]]
    l_cf_ds = [cf_ds[0], cf_ds[24], cf_ds[12], cf_ds[2], cf_ds[26], cf_ds[14], cf_ds[1], cf_ds[25], cf_ds[13]]
    l_cf_params = [cf_params[0], cf_params[24], cf_params[12], cf_params[2], cf_params[26], cf_params[14], cf_params[1], cf_params[25], cf_params[13]]
    cf_block = fill_block_exp2_long(l_cf, l_cf_params, l_cf_ds)

    col = df['mse_gradient_descent']
    col_ds = df_ds['mse_gradient_descent']
    col_params = df_params['mse_gradient_descent']
    blocks = []
    for i in range(4):
        l = [col[0 + i * 3], col[24 + i * 3], col[12 + i * 3], col[2 + i * 3], col[26 + i * 3], col[14 + i * 3],
             col[1 + i * 3], col[25 + i * 3], col[13 + i * 3]]
        l_ds = [col_ds[0 + i * 3], col_ds[24 + i * 3], col_ds[12 + i * 3], col_ds[2 + i * 3], col_ds[26 + i * 3], col_ds[14 + i * 3],
             col_ds[1 + i * 3], col_ds[25 + i * 3], col_ds[13 + i * 3]]
        l_params = [col_params[0 + i * 3], col_params[24 + i * 3], col_params[12 + i * 3], col_params[2 + i * 3], col_params[26 + i * 3], col_params[14 + i * 3],
             col_params[1 + i * 3], col_params[25 + i * 3], col_params[13 + i * 3]]
        block = fill_block_exp2_long(l, l_params, l_ds)
        blocks.append(block)
    [rev_kl_block, fwd_kl_block, mle_ds_block, mle_pa_block] = blocks

    out = rf"""\begin{{table*}}[h!]
    \centering
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.2}}
    \begin{{tabular}}{{lllccccccccc}}
    \toprule
    \textbf{{Model Family}} & \textbf {{Objective}} & \textbf {{Modelling Assumption}} & \multicolumn {{9}}{{c}}{{$L_2$ Loss($\downarrow$)}} \\
    \cmidrule(lr){{4 - 12}}
    & & & \multicolumn{{3}}{{c}}{{Linear}} & \multicolumn {{3}}{{c}}{{Polynomial}} & \multicolumn{{3}}{{c}}{{Nonlinear}} \\
    \cmidrule(lr){{4 - 6}}\cmidrule(lr){{7 - 9}}\cmidrule(lr){{10 - 12}} & & & Default & Inc.Parameters & Inc.Dataset Size & Default & Inc.Parameters & Inc.Dataset Size & Default & Inc.Parameters & Inc.Dataset Size \\
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \multirow{{3}}{{*}}{{Closed Form Solution}} & \multirow{{3}}{{*}}{{MLE}}\
    {cf_block}
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}\multirow{{12}}{{*}}{{Amortized}} & \multirow{{3}}{{*}}{{Rev-KL}}
    {rev_kl_block}
    \cmidrule{{2 - 12}}
    &  \multirow{{3}}{{*}}{{Fwd-KL}}
    {fwd_kl_block}
    \cmidrule{{2 - 12}}
    &  \multirow{{3}}{{*}}{{MLE-Dataset}}
    {mle_ds_block}
    \cmidrule{{2 - 12}}
    &  \multirow{{3}}{{*}}{{MLE-Params}}
    {mle_pa_block}
    \end{{tabular}}\caption{{}}\end{{table*}}
    """
    print(out)

def fill_block_exp2_kl(l):
    string = rf"""& Linear & {l[0]:.5f} & {l[1]:.5f} & {l[2]:.5f} & {l[3]:.5f} & {l[4]:.5f} & {l[5]:.5f} \\
    & & Polynomial & {l[6]:.5f} & {l[7]:.5f} & {l[8]:.5f} & {l[9]:.5f} & {l[10]:.5f} & {l[11]:.5f} \\
    & & Nonlinear & {l[12]:.5f} & {l[13]:.5f} & {l[14]:.5f} & {l[15]:.5f} & {l[16]:.5f} & {l[17]:.5f} \\
    """
    return string

def convert_exp2_kl(filename):
    df = pd.read_csv(filename)
    b_fwd = df['baseline_fwd']
    b_rev = df['baseline_rev']
    rev_kl = df['bw_kl']
    fwd_kl = df['fw_kl']

    prior_block = fill_block_exp2_kl([b_fwd[0], b_rev[0], b_fwd[24], b_rev[24], b_fwd[12], b_rev[12], b_fwd[2], b_rev[2], b_fwd[26], b_rev[26], b_fwd[14], b_rev[14], b_fwd[1], b_rev[1], b_fwd[25], b_rev[25], b_fwd[13], b_rev[13]])
    blocks = []
    for i in range(2):
        l = fill_block_exp2_kl([fwd_kl[0+i*3], rev_kl[0+i*3], fwd_kl[24+i*3], rev_kl[24+i*3], fwd_kl[12+i*3], rev_kl[12+i*3], fwd_kl[2+i*3], rev_kl[2+i*3], fwd_kl[26+i*3], rev_kl[26+i*3], fwd_kl[14+i*3], rev_kl[14+i*3], fwd_kl[1+i*3], rev_kl[1+i*3], fwd_kl[25+i*3], rev_kl[25+i*3], fwd_kl[13+i*3], rev_kl[13+i*3]])
        blocks.append(l)
    [rev_kl_block, fwd_kl_block] = blocks
    out = rf"""
    \begin{{table*}}[h!]
    \centering
    \small
    \setlength{{\tabcolsep}}{{6pt}}
    \renewcommand{{\arraystretch}}{{1.2}}
    \resizebox{{\textwidth}}{{!}}{{ 
    \begin{{tabular}}{{lllcccccc}}
    \toprule
    \textbf{{Model Family}} & \textbf{{Objective}} & \textbf{{Modelling Assumption}} & \multicolumn{{6}}{{c}}{{\$L_2\$ Loss (\$\downarrow\$)}} \\
    \cmidrule(lr){{4-9}}
    & & &
    \multicolumn{{2}}{{c}}{{Linear}} &
    \multicolumn{{2}}{{c}}{{Polynomial}} &
    \multicolumn{{2}}{{c}}{{Nonlinear}} \\
    \cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}\cmidrule(lr){{8-9}}
    & & & Fwd-KL & Rev-KL & Fwd-KL & Rev-KL & Fwd-KL & Rev-KL \\
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}

    \multirow{{3}}{{*}}{{Baseline}}
    & \multirow{{3}}{{*}}{{Prior}}
    {prior_block}
    \specialrule{{0.1em}}{{0.1em}}{{0.1em}}
    \multirow{{6}}{{*}}{{Amortized}}
    & \multirow{{3}}{{*}}{{Rev-KL}}
    {rev_kl_block}
    \cmidrule{{2-9}}
    & \multirow{{3}}{{*}}{{Fwd-KL}}
    {fwd_kl_block}
    \cmidrule{{2-9}}
    \end{{tabular}}}}
    \caption{{}} 
    \end{{table*}}
    """

    print(out)

convert_exp2_ablation_eb("./exp2_uniform_fixed_no_normalize_final/experiment2_results.csv", "./exp2_uniform_fixed_no_normalize_inc_ds_size/experiment2_results.csv", "./exp2_uniform_fixed_no_normalize_inc_params/experiment2_results.csv")
#convert_exp2_kl("./exp2_uniform_fixed_no_normalize_final/experiment2_results.csv")
#convert_exp2_eb("./exp2_uniform_fixed_no_normalize_final/experiment2_results.csv")
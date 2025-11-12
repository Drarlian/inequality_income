import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from pathlib import Path


def to_numeric_brl(s: pd.Series) -> pd.Series:
    """
    Converte strings numéricas com possíveis separadores BR (milhar . e decimal ,) para float.
    :param s:
    :return:
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def gini(values: np.ndarray) -> float:
    """
    Índice de Gini para um array 1D. Retorna 0 (igualdade perfeita) a 1 (desigualdade máxima).
    :param values:
    :return:
    """
    x = np.asarray(values, dtype=float).flatten()
    if x.size == 0:
        return float("nan")
    if np.any(x < 0):
        x = x - x.min() + 1e-12
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def simulate_population_from_means(means: np.ndarray, cv: float = 0.5, samples_per_row: int = 500, seed: int = 2025) -> np.ndarray:
    """
    Gera população sintética lognormal com média aproximada por linha (CV fixo).
    :param means:
    :param cv:
    :param samples_per_row:
    :param seed:
    :return:
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.log(1 + cv**2)
    sigma = np.sqrt(sigma2)
    simulated = []
    for m in means:
        if not np.isfinite(m) or m <= 0:
            continue
        mu = np.log(m) - 0.5 * sigma2  # E[lognormal] ≈ m
        simulated.append(rng.lognormal(mean=mu, sigma=sigma, size=samples_per_row))
    if not simulated:
        return np.array([], dtype=float)
    return np.concatenate(simulated)


def read_file(csv_path: str = "renda_educacao.csv") -> pd.DataFrame:
    """
    Le e limpa os dados do csv.
    :param csv_path: Path do arqyivo csv
    :return: Retorna um dataframe pandas limpo.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    expected = {"escolaridade", "renda_media", "regiao"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}. Encontradas: {list(df.columns)}")

    df["renda_media"] = to_numeric_brl(df["renda_media"])
    df["escolaridade"] = df["escolaridade"].astype(str).str.strip()
    df["regiao"] = df["regiao"].astype(str).str.strip()
    df = df.dropna(subset=["renda_media", "escolaridade", "regiao"]).copy()
    return df


def create_dispersao_escolaridade_renda(df_corr: pd.DataFrame, output_dir: Path) -> Path:
    """
    Cria um gráfico de dispersão de Escolaridade x Renda utilizando o Seaborn.
    :param df_corr:
    :param output_dir:
    :return: Path onde foi salvo o arquivo.
    """
    sns.set_theme(context="notebook", style="whitegrid")
    rng = np.random.default_rng(123)
    df_corr["escolaridade_code_j"] = df_corr["escolaridade_code"].astype(float) + rng.uniform(-0.1, 0.1, size=len(df_corr))

    ax = sns.scatterplot(
        data=df_corr,
        x="escolaridade_code_j",
        y="renda_media",
        s=40,
    )
    xticks_pos = sorted(df_corr["escolaridade_code"].unique())
    xticks_labels = (
        df_corr.sort_values("escolaridade_code")[["escolaridade_code", "escolaridade"]]
        .drop_duplicates()["escolaridade"]
        .tolist()
    )
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels, rotation=0)
    ax.set_xlabel("Escolaridade (categorias ordenadas)")
    ax.set_ylabel("Renda média")
    ax.set_title("Dispersão: Escolaridade x Renda média")
    fig_scatter = ax.get_figure()
    scatter_path = output_dir / "scatter_escolaridade_renda.png"
    fig_scatter.tight_layout()
    fig_scatter.savefig(scatter_path, dpi=150)
    plt.close(fig_scatter)

    return scatter_path


def create_dispersao_regiao_renda(df_corr: pd.DataFrame, output_dir: Path) -> Path:
    """
    Cria um gráfico de Dispersão de Região x Renda.
    Utiliza stripplot com jitter para enxergar a distribuição da renda por região.
    :return: None
    """
    ax = sns.stripplot(
        data=df_corr,
        x="regiao",
        y="renda_media",
        hue="escolaridade_cat",  # -> Ajuda a ver escolaridade dentro de cada região.
        dodge=True,
        jitter=True,
        alpha=0.8,
        size=4,
    )
    ax.set_xlabel("Região")
    ax.set_ylabel("Renda média")
    ax.set_title("Dispersão: Renda por Região (com escolaridade)")
    ax.legend(title="Escolaridade", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    scatter_regiao_path = output_dir / "scatter_renda_por_regiao.png"
    plt.savefig(scatter_regiao_path, dpi=150)
    plt.close()

    return scatter_regiao_path


def create_heatmap_renda_media_regiao_escolaridade(df_corr: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, Path]:
    """
    Cria um heatmap de Renda Média por Região x Escolaridade.
    :param df_corr:
    :param output_dir:
    :return: None
    """
    pivot = (
        df_corr.pivot_table(
            index="regiao",
            columns="escolaridade_cat",
            values="renda_media",
            aggfunc="mean",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Renda média"},
    )
    ax.set_title("Heatmap: Renda Média por Região x Escolaridade")
    ax.set_xlabel("Escolaridade")
    ax.set_ylabel("Região")
    fig_heat = ax.get_figure()
    heatmap_path = output_dir / "heatmap_renda_regiao.png"
    fig_heat.tight_layout()
    fig_heat.savefig(heatmap_path, dpi=150)
    plt.close(fig_heat)

    return pivot, heatmap_path


def calculate_indice_gini(df: pd.DataFrame, output_dir: Path) -> tuple[float, float, Path]:
    """
    Calcula o índice Gini e salva os resultados em um arquivo JSON.
    :param df:
    :param output_dir:
    :return:
    """
    # 6) Índice de Gini (empírico e simulado) -> salvar em JSON
    cv = 0.5
    samples_per_row = 500
    seed = 2025

    gini_empirico = gini(df["renda_media"].to_numpy())
    synthetic_pop = simulate_population_from_means(
        df["renda_media"].to_numpy(),
        cv=cv,
        samples_per_row=samples_per_row,
        seed=seed,
    )
    gini_simulado = gini(synthetic_pop)

    gini_json_path = output_dir / "gini_resultados.json"
    with open(gini_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gini_empirico": float(gini_empirico),
                "gini_simulado": float(gini_simulado),
                "parametros_simulacao": {
                    "cv": float(cv),
                    "samples_per_row": int(samples_per_row),
                    "seed": int(seed),
                },
                "n_registros": int(df.shape[0]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return gini_empirico ,gini_simulado, gini_json_path


def create_resume(df: pd.DataFrame, df_corr: pd.DataFrame, gini_empirico: float, gini_simulado: float, pivot: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """
    Cria um arquivo com o resumo das informações.
    :return: None
    """
    # Correlação (Pearson e Spearman):
    pearson = df_corr[["escolaridade_code", "renda_media"]].corr(method="pearson").iloc[0, 1]
    spearman = df_corr[["escolaridade_code", "renda_media"]].corr(method="spearman").iloc[0, 1]

    # Saídas tabulares:
    resumo = pd.DataFrame(
        {
            "Métrica": [
                "Correlação (Pearson) escolaridade x renda",
                "Correlação (Spearman) escolaridade x renda",
                "Gini (empírico) sobre renda_media",
                "Gini (simulado) com população sintética",
                "Registros usados (após limpeza)",
            ],
            "Valor": [
                round(float(pearson), 4) if pd.notna(pearson) else np.nan,
                round(float(spearman), 4) if pd.notna(spearman) else np.nan,
                round(float(gini_empirico), 4),
                round(float(gini_simulado), 4),
                int(df.shape[0]),
            ],
        }
    )

    pivot_csv = output_dir / "pivot_renda_regiao_escolaridade.csv"
    resumo_csv = output_dir / "resumo_metricas.csv"
    pivot.to_csv(pivot_csv, index=True)
    resumo.to_csv(resumo_csv, index=False)

    return pivot_csv, resumo_csv


def main(csv_path: str | Path = "renda_educacao.csv", output_dir: str | Path = "outputs") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_file(csv_path)

    # Codificação ordinal de escolaridade (para correlação numérica):
    education_order = [
        "Sem instrução",
        "Fundamental",
        "Médio",
        "Técnico",
        "Superior",
        "Pós-graduação",
        "Mestrado",
        "Doutorado",
    ]
    cat_type = CategoricalDtype(categories=education_order, ordered=True)
    df["escolaridade_cat"] = df["escolaridade"].astype(cat_type)
    df["escolaridade_code"] = df["escolaridade_cat"].cat.codes

    # Filtra apenas as linhas com escolaridade dentro da ordem definida (codes >= 0):
    df_corr = df.loc[df["escolaridade_code"] >= 0].copy()

    scatter_path = create_dispersao_escolaridade_renda(df_corr, output_dir)

    scatter_regiao_path = create_dispersao_regiao_renda(df_corr, output_dir)

    pivot, heatmap_path = create_heatmap_renda_media_regiao_escolaridade(df_corr, output_dir)

    gini_empirico ,gini_simulado, gini_json_path = calculate_indice_gini(df, output_dir)

    pivot_csv, resumo_csv = create_resume(df, df_corr, gini_empirico, gini_simulado, pivot, output_dir)

    # Arquivo com caminhos de saída:
    manifest = {
        "arquivos": {
            "scatter_escolaridade_renda_png": str(scatter_path),
            "scatter_renda_por_regiao_png": str(scatter_regiao_path),  # NOVO
            "heatmap_png": str(heatmap_path),
            "pivot_csv": str(pivot_csv),
            "resumo_csv": str(resumo_csv),
            "gini_json": str(gini_json_path),
        }
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

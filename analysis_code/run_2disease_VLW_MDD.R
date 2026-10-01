################################################################################
# 2 Mental Disorders — Value of Lost Welfare (VLW) Analysis, IE = 1.0
# GBD 2023 | LMIC countries | 0-19 years | 1990-2023
#
# Diseases: Depressive disorders (cause_id = 567)
#           Anxiety disorders    (cause_id = 571)
#
# Age groups: <5 yrs, 5-9 yrs, 10-14 yrs, 15-19 yrs, <20 yrs (aggregate)
#
# Outputs (results_VLW_MDD/2disease_IE1/):
#   CSVs : T1–T6, T_age_*
#   PDFs : F1–F6, F_age_*
################################################################################

library(tidyverse)
library(sf)
library(patchwork)
library(scales)

# ===========================================================================
# PARAMETERS
# ===========================================================================
VSL_peak_USA  <- 13.2e6      # USD, US VSL (EPA 2023)
GDP_pc_USA    <- 82304.62    # USD PPP, US GDP per capita 2023 (World Bank)
discount_rate <- 0.03        # 3% annual discount rate

args   <- commandArgs(trailingOnly = TRUE)
IE     <- if (length(args) >= 1) as.numeric(args[1]) else 1.0
ie_tag <- if (IE == 1.0) "IE1" else paste0("IE", IE)

# Input files (relative to working directory = MDD folder)
daly_file  <- "merged_gbd_data.csv"
hale_file  <- "HALE.csv"
gdp_file   <- "gdp.csv"
lmic_file  <- "204_with_LMIC.csv"
world_file <- "df_world2.geojson"

# Output directory
out_dir <- file.path("results_VLW_MDD", paste0("2disease_", ie_tag))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ===========================================================================
# DISEASE & INCOME LABELS / COLOURS
# ===========================================================================
disease_map <- c(
  "Depressive disorders" = "Depressive disorders",
  "Anxiety disorders"    = "Anxiety disorders"
)
disease_levels <- c("Depressive disorders", "Anxiety disorders")

disease_colors <- c(
  "Depressive disorders" = "#1B4F72",
  "Anxiety disorders"    = "#C0392B"
)

disease_palettes <- list(
  "Depressive disorders" = c("#EFF7FF","#C6DBEF","#9ECAE1","#4292C6","#2171B5","#084594"),
  "Anxiety disorders"    = c("#FFF5F0","#FCBBA1","#FB6A4A","#EF3B2C","#CB181D","#67000D")
)

income_levels <- c("Low income", "Lower middle income", "Upper middle income")
income_colors <- c(
  "Low income"          = "#C0392B",
  "Lower middle income" = "#E67E22",
  "Upper middle income" = "#117A65"
)

# Age group ordering (exclude <20 aggregate from age-panel figures; use for totals)
age_levels <- c("<5 years", "5-9 years", "10-14 years", "15-19 years")
age_labels <- c("0-4", "5-9", "10-14", "15-19")

# GBD → World Bank country name mapping
gbd_to_wb <- c(
  "Bolivia (Plurinational State of)"      = "Bolivia",
  "Congo"                                 = "Congo, Rep.",
  "Côte d'Ivoire"                         = "Cote d'Ivoire",
  "Democratic People's Republic of Korea" = "Korea, Dem. People's Rep.",
  "Democratic Republic of the Congo"      = "Congo, Dem. Rep.",
  "Egypt"                                 = "Egypt, Arab Rep.",
  "Gambia"                                = "Gambia, The",
  "Iran (Islamic Republic of)"            = "Iran, Islamic Rep.",
  "Kyrgyzstan"                            = "Kyrgyz Republic",
  "Lao People's Democratic Republic"      = "Lao PDR",
  "Micronesia (Federated States of)"      = "Micronesia, Fed. Sts.",
  "Palestine"                             = "West Bank and Gaza",
  "Republic of Moldova"                   = "Moldova",
  "Saint Kitts and Nevis"                 = "St. Kitts and Nevis",
  "Saint Lucia"                           = "St. Lucia",
  "Saint Vincent and the Grenadines"      = "St. Vincent and the Grenadines",
  "Somalia"                               = "Somalia, Fed. Rep.",
  "Türkiye"                               = "Turkiye",
  "United Republic of Tanzania"           = "Tanzania",
  "Venezuela (Bolivarian Republic of)"    = "Venezuela, RB",
  "Viet Nam"                              = "Viet Nam",
  "Yemen"                                 = "Yemen, Rep."
)

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================
norm_name <- function(x) {
  x %>%
    str_replace_all("\u2019", "'") %>%
    str_replace_all("\u2018", "'") %>%
    str_replace_all("\u00a0", " ") %>%
    iconv(from = "UTF-8", to = "ASCII//TRANSLIT") %>%
    str_squish()
}

fmt_ui <- function(val, lower, upper, digits = 1, scale = 1) {
  v <- val / scale; l <- lower / scale; u <- upper / scale
  paste0(
    formatC(v, format = "f", digits = digits, big.mark = ","),
    " (", formatC(l, format = "f", digits = digits, big.mark = ","),
    "\u2013", formatC(u, format = "f", digits = digits, big.mark = ","), ")"
  )
}

calc_discount <- function(remaining_years, r = 0.03) {
  ifelse(remaining_years <= 0, 0,
         (1 - (1 / (1 + r))^remaining_years) / (r * remaining_years))
}

theme_nm <- function(base_size = 8) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title       = element_blank(),
      axis.title       = element_text(size = base_size, face = "bold", colour = "grey20"),
      axis.text        = element_text(size = base_size - 1, colour = "grey30"),
      legend.title     = element_text(size = base_size - 1, face = "bold"),
      legend.text      = element_text(size = base_size - 1.5),
      legend.key.size  = unit(0.35, "cm"),
      panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
      panel.grid.minor = element_blank(),
      strip.text       = element_text(size = base_size, face = "bold"),
      plot.margin      = margin(4, 4, 4, 4, "pt")
    )
}

theme_nm_map <- function(base_size = 8) {
  theme_void(base_size = base_size) +
    theme(
      legend.position = "right",
      legend.title    = element_text(size = base_size - 1, face = "bold"),
      legend.text     = element_text(size = base_size - 1.5),
      legend.key.size = unit(0.35, "cm"),
      plot.margin     = margin(2, 2, 2, 2, "pt")
    )
}

# ===========================================================================
cat("══════════════════════════════════════════════════════════════════\n")
cat(sprintf("  MDD: Depressive & Anxiety Disorders VLW (0-19y) — IE = %.1f\n", IE))
cat("══════════════════════════════════════════════════════════════════\n\n")

# ===========================================================================
# 1. READ DATA
# ===========================================================================
cat("[1/5] Reading data...\n")

df_lmic <- read_csv(lmic_file, show_col_types = FALSE) %>%
  filter(LMIC == 1) %>%
  select(location_id, location_name, LMIC_group) %>%
  mutate(
    wb_country = case_when(
      location_name %in% names(gbd_to_wb) ~ gbd_to_wb[location_name],
      TRUE ~ location_name
    ),
    wb_norm = norm_name(wb_country)
  )

df_gdp_raw <- read_csv(gdp_file, show_col_types = FALSE) %>%
  filter(year == 2023, !is.na(NY.GDP.PCAP.PP.CD), !is.na(NY.GDP.MKTP.PP.CD)) %>%
  select(country,
         GDP_pc_PPP    = NY.GDP.PCAP.PP.CD,
         GDP_PPP_total = NY.GDP.MKTP.PP.CD) %>%
  mutate(country_norm = norm_name(country))

df_gdp <- df_lmic %>%
  left_join(df_gdp_raw, by = c("wb_norm" = "country_norm")) %>%
  filter(!is.na(GDP_pc_PPP)) %>%
  select(location_id, location_name, LMIC_group, GDP_pc_PPP, GDP_PPP_total)

lmic_ids <- df_gdp$location_id
cat("   LMIC countries with GDP data:", length(lmic_ids), "\n")

# DALYs — 2 diseases, <20 years (aggregate), both+male+female, 1990-2023
cat("   Loading DALYs (age <20, aggregate)...\n")
df_daly_agg <- read_csv(daly_file, show_col_types = FALSE) %>%
  filter(
    cause_name   %in% names(disease_map),
    measure_name == "DALYs (Disability-Adjusted Life Years)",
    metric_name  == "Number",
    age_name     == "<20 years",
    location_id  %in% lmic_ids
  ) %>%
  mutate(disease = disease_map[cause_name]) %>%
  select(location_id, location_name, sex_id, sex_name, age_name,
         disease, cause_name, year, DALY = val, lower, upper)

cat("   DALYs rows (agg):", nrow(df_daly_agg), "\n")
cat("   Diseases found:", paste(unique(df_daly_agg$disease), collapse = ", "), "\n")

# DALYs — age-stratified: <5, 5-9, 10-14, 15-19
cat("   Loading DALYs (4 age bands)...\n")
df_daly_age <- read_csv(daly_file, show_col_types = FALSE) %>%
  filter(
    cause_name   %in% names(disease_map),
    measure_name == "DALYs (Disability-Adjusted Life Years)",
    metric_name  == "Number",
    age_name     %in% age_levels,
    location_id  %in% lmic_ids
  ) %>%
  mutate(
    disease  = disease_map[cause_name],
    age_name = factor(age_name, levels = age_levels)
  ) %>%
  select(location_id, location_name, sex_id, sex_name, age_name,
         disease, year, DALY = val, lower, upper)

cat("   DALYs rows (age bands):", nrow(df_daly_age), "\n")

# HALE — All ages, both+male+female, 2023
df_hale <- read_csv(hale_file, show_col_types = FALSE) %>%
  filter(metric_name == "Years", year == 2023, age_name == "All ages") %>%
  select(location_id, sex_id, HALE = val)

# World map geometry
df_world <- st_read(world_file, quiet = TRUE)
df_world$location_id <- as.numeric(df_world$location_id)

cat("   All data loaded.\n\n")

# ===========================================================================
# 2. COMPUTE VLW  (aggregate <20 years)
# ===========================================================================
cat(sprintf("[2/5] Computing VLW (IE = %.1f, age <20 aggregate)...\n", IE))

df_vlw <- df_daly_agg %>%
  left_join(df_gdp,  by = c("location_id", "location_name")) %>%
  left_join(df_hale, by = c("location_id", "sex_id")) %>%
  filter(!is.na(GDP_pc_PPP), !is.na(HALE)) %>%
  mutate(
    disease    = factor(disease,    levels = disease_levels),
    LMIC_group = factor(LMIC_group, levels = income_levels),
    VSL_i         = VSL_peak_USA * (GDP_pc_PPP / GDP_pc_USA)^IE,
    VSLY          = VSL_i / (HALE / 2),
    remaining_yrs = HALE * 0.4,
    disc_factor   = calc_discount(remaining_yrs, discount_rate),
    VLW           = VSLY * DALY  / 1e9,
    VLW_lower     = VSLY * lower / 1e9,
    VLW_upper     = VSLY * upper / 1e9,
    VLW_disc      = VLW       * disc_factor,
    VLW_disc_lower = VLW_lower * disc_factor,
    VLW_disc_upper = VLW_upper * disc_factor,
    VLW_GDP_pct       = VLW       * 1e9 / GDP_PPP_total * 100,
    VLW_GDP_pct_lower = VLW_lower * 1e9 / GDP_PPP_total * 100,
    VLW_GDP_pct_upper = VLW_upper * 1e9 / GDP_PPP_total * 100,
    VLW_disc_GDP_pct       = VLW_disc       * 1e9 / GDP_PPP_total * 100,
    VLW_disc_GDP_pct_lower = VLW_disc_lower * 1e9 / GDP_PPP_total * 100,
    VLW_disc_GDP_pct_upper = VLW_disc_upper * 1e9 / GDP_PPP_total * 100
  )

n_countries <- n_distinct(df_vlw$location_id)
n_years     <- n_distinct(df_vlw$year)
cat(sprintf("   VLW computed: %d countries × %d years × 2 diseases × 3 sexes\n",
            n_countries, n_years))

df_2023 <- df_vlw %>% filter(year == 2023, sex_name == "Both")

# ===========================================================================
# 2b. COMPUTE VLW  (age-stratified: <5, 5-9, 10-14, 15-19)
# ===========================================================================
cat("   Computing VLW (age-stratified)...\n")

df_vlw_age <- df_daly_age %>%
  left_join(df_gdp,  by = c("location_id", "location_name")) %>%
  left_join(df_hale, by = c("location_id", "sex_id")) %>%
  filter(!is.na(GDP_pc_PPP), !is.na(HALE)) %>%
  mutate(
    disease    = factor(disease,    levels = disease_levels),
    LMIC_group = factor(LMIC_group, levels = income_levels),
    age_name   = factor(age_name,   levels = age_levels),
    VSL_i         = VSL_peak_USA * (GDP_pc_PPP / GDP_pc_USA)^IE,
    VSLY          = VSL_i / (HALE / 2),
    remaining_yrs = HALE * 0.4,
    disc_factor   = calc_discount(remaining_yrs, discount_rate),
    VLW           = VSLY * DALY  / 1e9,
    VLW_lower     = VSLY * lower / 1e9,
    VLW_upper     = VSLY * upper / 1e9,
    VLW_disc      = VLW       * disc_factor,
    VLW_disc_lower = VLW_lower * disc_factor,
    VLW_disc_upper = VLW_upper * disc_factor,
    VLW_GDP_pct       = VLW       * 1e9 / GDP_PPP_total * 100,
    VLW_GDP_pct_lower = VLW_lower * 1e9 / GDP_PPP_total * 100,
    VLW_GDP_pct_upper = VLW_upper * 1e9 / GDP_PPP_total * 100
  )

df_age_2023 <- df_vlw_age %>% filter(year == 2023)

cat("   Done.\n\n")

# ===========================================================================
# 3. TABLES (T1–T6 + T_age)
# ===========================================================================
cat("[3/5] Saving tables...\n")

# ── T1: Country × Disease × Sex — 2023 ──────────────────────────────────
tbl_country <- df_vlw %>%
  filter(year == 2023) %>%
  mutate(
    `DALYs`                        = fmt_ui(DALY, lower, upper, 0),
    `VLW (billion USD)`            = fmt_ui(VLW, VLW_lower, VLW_upper, 3),
    `VLW discounted (billion USD)` = fmt_ui(VLW_disc, VLW_disc_lower, VLW_disc_upper, 3),
    `VLW/GDP (%)`                  = fmt_ui(VLW_GDP_pct, VLW_GDP_pct_lower, VLW_GDP_pct_upper, 3),
    `VLW discounted/GDP (%)`       = fmt_ui(VLW_disc_GDP_pct, VLW_disc_GDP_pct_lower, VLW_disc_GDP_pct_upper, 3)
  ) %>%
  transmute(
    Country = location_name, Income_Group = LMIC_group, Sex = sex_name,
    Disease = disease, Age = age_name,
    GDP_pc_PPP = round(GDP_pc_PPP), GDP_total_bn = round(GDP_PPP_total / 1e9, 2),
    `DALYs`, `VLW (billion USD)`, `VLW discounted (billion USD)`,
    `VLW/GDP (%)`, `VLW discounted/GDP (%)`
  )
write_csv(tbl_country, file.path(out_dir, "T1_country_disease_2023.csv"))
cat("   T1 saved.\n")

# ── T2: Disease × Income group summary (Both sexes, 2023) ────────────────
tbl_disease_income <- df_2023 %>%
  group_by(disease, LMIC_group) %>%
  summarise(
    n_countries       = n_distinct(location_id),
    DALY_total        = sum(DALY),     DALY_lower        = sum(lower),           DALY_upper        = sum(upper),
    VLW_total         = sum(VLW),      VLW_total_lower   = sum(VLW_lower),       VLW_total_upper   = sum(VLW_upper),
    VLW_disc_total    = sum(VLW_disc), VLW_disc_lower    = sum(VLW_disc_lower),  VLW_disc_upper    = sum(VLW_disc_upper),
    VLW_GDP_wtd       = weighted.mean(VLW_GDP_pct,       GDP_PPP_total),
    VLW_GDP_wtd_lower = weighted.mean(VLW_GDP_pct_lower, GDP_PPP_total),
    VLW_GDP_wtd_upper = weighted.mean(VLW_GDP_pct_upper, GDP_PPP_total),
    .groups = "drop"
  ) %>%
  mutate(
    `DALYs (thousands)`       = fmt_ui(DALY_total,     DALY_lower,        DALY_upper,        1, 1000),
    `VLW (billion USD)`       = fmt_ui(VLW_total,      VLW_total_lower,   VLW_total_upper,   2),
    `VLW discounted (bn USD)` = fmt_ui(VLW_disc_total, VLW_disc_lower,    VLW_disc_upper,    2),
    `VLW/GDP (%)`             = fmt_ui(VLW_GDP_wtd,    VLW_GDP_wtd_lower, VLW_GDP_wtd_upper, 3)
  ) %>%
  arrange(disease, LMIC_group) %>%
  select(Disease = disease, `Income Group` = LMIC_group, Countries = n_countries,
         `DALYs (thousands)`, `VLW (billion USD)`, `VLW discounted (bn USD)`, `VLW/GDP (%)`)
write_csv(tbl_disease_income, file.path(out_dir, "T2_disease_income_summary_2023.csv"))
cat("   T2 saved.\n")

# ── T3: Disease summary all LMICs (Both sexes, 2023) ─────────────────────
tbl_disease_total <- df_2023 %>%
  group_by(disease) %>%
  summarise(
    n_countries       = n_distinct(location_id),
    DALY_total        = sum(DALY),     DALY_lower        = sum(lower),           DALY_upper        = sum(upper),
    VLW_total         = sum(VLW),      VLW_total_lower   = sum(VLW_lower),       VLW_total_upper   = sum(VLW_upper),
    VLW_disc_total    = sum(VLW_disc), VLW_disc_lower    = sum(VLW_disc_lower),  VLW_disc_upper    = sum(VLW_disc_upper),
    VLW_GDP_wtd       = weighted.mean(VLW_GDP_pct,       GDP_PPP_total),
    VLW_GDP_wtd_lower = weighted.mean(VLW_GDP_pct_lower, GDP_PPP_total),
    VLW_GDP_wtd_upper = weighted.mean(VLW_GDP_pct_upper, GDP_PPP_total),
    .groups = "drop"
  ) %>%
  mutate(
    VLW_share_pct             = VLW_total / sum(VLW_total) * 100,
    `DALYs (thousands)`       = fmt_ui(DALY_total,     DALY_lower,        DALY_upper,        1, 1000),
    `VLW (billion USD)`       = fmt_ui(VLW_total,      VLW_total_lower,   VLW_total_upper,   2),
    `VLW discounted (bn USD)` = fmt_ui(VLW_disc_total, VLW_disc_lower,    VLW_disc_upper,    2),
    `VLW/GDP (%)`             = fmt_ui(VLW_GDP_wtd,    VLW_GDP_wtd_lower, VLW_GDP_wtd_upper, 3),
    `Share of total VLW (%)`  = round(VLW_share_pct, 1)
  ) %>%
  arrange(desc(VLW_total)) %>%
  select(Disease = disease, Countries = n_countries,
         `DALYs (thousands)`, `VLW (billion USD)`, `VLW discounted (bn USD)`,
         `VLW/GDP (%)`, `Share of total VLW (%)`)
write_csv(tbl_disease_total, file.path(out_dir, "T3_disease_summary_allLMIC_2023.csv"))
cat("   T3 saved.\n")

# ── T4: Combined 2-disease total by income group ──────────────────────────
tbl_combined <- df_2023 %>%
  group_by(location_id, location_name, LMIC_group, GDP_pc_PPP, GDP_PPP_total) %>%
  summarise(
    VLW_combined            = sum(VLW),         VLW_combined_lower  = sum(VLW_lower),        VLW_combined_upper  = sum(VLW_upper),
    VLW_disc_combined       = sum(VLW_disc),     VLW_disc_comb_lower = sum(VLW_disc_lower),   VLW_disc_comb_upper = sum(VLW_disc_upper),
    DALY_combined           = sum(DALY),         DALY_combined_lower = sum(lower),            DALY_combined_upper = sum(upper),
    .groups = "drop"
  ) %>%
  mutate(
    VLW_comb_GDP_pct            = VLW_combined            * 1e9 / GDP_PPP_total * 100,
    VLW_comb_GDP_pct_lower      = VLW_combined_lower      * 1e9 / GDP_PPP_total * 100,
    VLW_comb_GDP_pct_upper      = VLW_combined_upper      * 1e9 / GDP_PPP_total * 100,
    VLW_disc_comb_GDP_pct       = VLW_disc_combined       * 1e9 / GDP_PPP_total * 100,
    VLW_disc_comb_GDP_pct_lower = VLW_disc_comb_lower     * 1e9 / GDP_PPP_total * 100,
    VLW_disc_comb_GDP_pct_upper = VLW_disc_comb_upper     * 1e9 / GDP_PPP_total * 100
  )

tbl_combined_income <- tbl_combined %>%
  group_by(LMIC_group) %>%
  summarise(
    n_countries  = n(),
    VLW_total    = sum(VLW_combined),        VLW_lower       = sum(VLW_combined_lower),    VLW_upper       = sum(VLW_combined_upper),
    VLW_disc     = sum(VLW_disc_combined),   VLW_disc_lower  = sum(VLW_disc_comb_lower),   VLW_disc_upper  = sum(VLW_disc_comb_upper),
    DALY_total   = sum(DALY_combined),       DALY_lower      = sum(DALY_combined_lower),   DALY_upper      = sum(DALY_combined_upper),
    VLW_GDP_wtd       = weighted.mean(VLW_comb_GDP_pct,       GDP_PPP_total),
    VLW_GDP_wtd_lower = weighted.mean(VLW_comb_GDP_pct_lower, GDP_PPP_total),
    VLW_GDP_wtd_upper = weighted.mean(VLW_comb_GDP_pct_upper, GDP_PPP_total),
    .groups = "drop"
  )

tbl_combined_total <- tbl_combined %>%
  summarise(
    LMIC_group   = "All LMICs",
    n_countries  = n(),
    VLW_total    = sum(VLW_combined),        VLW_lower       = sum(VLW_combined_lower),    VLW_upper       = sum(VLW_combined_upper),
    VLW_disc     = sum(VLW_disc_combined),   VLW_disc_lower  = sum(VLW_disc_comb_lower),   VLW_disc_upper  = sum(VLW_disc_comb_upper),
    DALY_total   = sum(DALY_combined),       DALY_lower      = sum(DALY_combined_lower),   DALY_upper      = sum(DALY_combined_upper),
    VLW_GDP_wtd       = weighted.mean(VLW_comb_GDP_pct,       GDP_PPP_total),
    VLW_GDP_wtd_lower = weighted.mean(VLW_comb_GDP_pct_lower, GDP_PPP_total),
    VLW_GDP_wtd_upper = weighted.mean(VLW_comb_GDP_pct_upper, GDP_PPP_total)
  )

tbl4 <- bind_rows(tbl_combined_income, tbl_combined_total) %>%
  mutate(
    LMIC_group                = factor(LMIC_group, levels = c(income_levels, "All LMICs")),
    `DALYs (thousands)`       = fmt_ui(DALY_total,  DALY_lower,        DALY_upper,        1, 1000),
    `VLW (billion USD)`       = fmt_ui(VLW_total,   VLW_lower,         VLW_upper,         2),
    `VLW discounted (bn USD)` = fmt_ui(VLW_disc,    VLW_disc_lower,    VLW_disc_upper,    2),
    `VLW/GDP (%)`             = fmt_ui(VLW_GDP_wtd, VLW_GDP_wtd_lower, VLW_GDP_wtd_upper, 3)
  ) %>%
  arrange(LMIC_group) %>%
  select(`Income Group` = LMIC_group, Countries = n_countries,
         `DALYs (thousands)`, `VLW (billion USD)`, `VLW discounted (bn USD)`, `VLW/GDP (%)`)
write_csv(tbl4, file.path(out_dir, "T4_combined2_by_income_2023.csv"))
cat("   T4 saved.\n")

# ── T5: Temporal trends ───────────────────────────────────────────────────
trend_combined_income <- df_vlw %>%
  filter(sex_name == "Both") %>%
  group_by(year, LMIC_group) %>%
  summarise(
    VLW_total  = sum(VLW),  VLW_lower  = sum(VLW_lower),  VLW_upper  = sum(VLW_upper),
    DALY_total = sum(DALY), DALY_lower = sum(lower),       DALY_upper = sum(upper),
    .groups = "drop"
  )

trend_by_disease <- df_vlw %>%
  filter(sex_name == "Both") %>%
  group_by(year, disease) %>%
  summarise(
    VLW_total  = sum(VLW),  VLW_lower  = sum(VLW_lower),  VLW_upper  = sum(VLW_upper),
    DALY_total = sum(DALY), DALY_lower = sum(lower),       DALY_upper = sum(upper),
    .groups = "drop"
  )

write_csv(trend_combined_income, file.path(out_dir, "T5a_trend_combined_by_income.csv"))
write_csv(trend_by_disease,      file.path(out_dir, "T5b_trend_by_disease.csv"))
cat("   T5 saved (T5a: by income; T5b: by disease).\n")

# ── T6: Country-level combined 2-disease (Both, 2023) ─────────────────────
tbl_country_combined <- tbl_combined %>% arrange(desc(VLW_comb_GDP_pct))

tbl6 <- tbl_country_combined %>%
  mutate(
    `DALYs (2 diseases)`           = fmt_ui(DALY_combined,     DALY_combined_lower,     DALY_combined_upper,     0),
    `VLW (billion USD)`            = fmt_ui(VLW_combined,      VLW_combined_lower,      VLW_combined_upper,      3),
    `VLW discounted (billion USD)` = fmt_ui(VLW_disc_combined, VLW_disc_comb_lower,     VLW_disc_comb_upper,     3),
    `VLW/GDP (%)`                  = fmt_ui(VLW_comb_GDP_pct,  VLW_comb_GDP_pct_lower,  VLW_comb_GDP_pct_upper,  3),
    `VLW discounted/GDP (%)`       = fmt_ui(VLW_disc_comb_GDP_pct, VLW_disc_comb_GDP_pct_lower, VLW_disc_comb_GDP_pct_upper, 3)
  ) %>%
  transmute(
    Country = location_name, Income_Group = LMIC_group,
    GDP_pc_PPP = round(GDP_pc_PPP), GDP_total_bn = round(GDP_PPP_total / 1e9, 2),
    `DALYs (2 diseases)`, `VLW (billion USD)`, `VLW discounted (billion USD)`,
    `VLW/GDP (%)`, `VLW discounted/GDP (%)`
  )
write_csv(tbl6, file.path(out_dir, "T6_country_combined2_2023.csv"))
cat("   T6 saved.\n")

# ── T_age: Age-stratified tables (2023) ──────────────────────────────────
# Age × Disease × Sex
tbl_age_disease_sex <- df_age_2023 %>%
  group_by(age_name, disease, sex_name) %>%
  summarise(
    n_countries  = n_distinct(location_id),
    DALY_total   = sum(DALY),     DALY_lower   = sum(lower),          DALY_upper   = sum(upper),
    VLW_total    = sum(VLW),      VLW_lower    = sum(VLW_lower),      VLW_upper    = sum(VLW_upper),
    VLW_GDP_wtd       = weighted.mean(VLW_GDP_pct,       GDP_PPP_total),
    VLW_GDP_wtd_lower = weighted.mean(VLW_GDP_pct_lower, GDP_PPP_total),
    VLW_GDP_wtd_upper = weighted.mean(VLW_GDP_pct_upper, GDP_PPP_total),
    .groups = "drop"
  ) %>%
  mutate(
    `DALYs`             = fmt_ui(DALY_total, DALY_lower,       DALY_upper,       0),
    `VLW (billion USD)` = fmt_ui(VLW_total,  VLW_lower,        VLW_upper,        3),
    `VLW/GDP (%)`       = fmt_ui(VLW_GDP_wtd, VLW_GDP_wtd_lower, VLW_GDP_wtd_upper, 3)
  ) %>%
  arrange(age_name, disease, sex_name) %>%
  select(`Age Group` = age_name, Disease = disease, Sex = sex_name,
         `DALYs`, `VLW (billion USD)`, `VLW/GDP (%)`)

write_csv(tbl_age_disease_sex, file.path(out_dir, "T_age_by_disease_sex_2023.csv"))

# Age × combined
tbl_age_combined <- df_age_2023 %>%
  filter(sex_name == "Both") %>%
  group_by(age_name) %>%
  summarise(
    DALY_total = sum(DALY),     DALY_lower = sum(lower),     DALY_upper = sum(upper),
    VLW_total  = sum(VLW),      VLW_lower  = sum(VLW_lower), VLW_upper  = sum(VLW_upper),
    VLW_GDP_wtd       = weighted.mean(VLW_GDP_pct,       GDP_PPP_total),
    VLW_GDP_wtd_lower = weighted.mean(VLW_GDP_pct_lower, GDP_PPP_total),
    VLW_GDP_wtd_upper = weighted.mean(VLW_GDP_pct_upper, GDP_PPP_total),
    .groups = "drop"
  ) %>%
  mutate(
    `DALYs`             = fmt_ui(DALY_total, DALY_lower,       DALY_upper,        0),
    `VLW (billion USD)` = fmt_ui(VLW_total,  VLW_lower,        VLW_upper,         3),
    `VLW/GDP (%)`       = fmt_ui(VLW_GDP_wtd, VLW_GDP_wtd_lower, VLW_GDP_wtd_upper, 3)
  ) %>%
  arrange(age_name) %>%
  select(`Age Group` = age_name, `DALYs`, `VLW (billion USD)`, `VLW/GDP (%)`)

write_csv(tbl_age_combined, file.path(out_dir, "T_age_combined_2023.csv"))
cat("   T_age saved.\n\n")

# ===========================================================================
# 4. FIGURES
# ===========================================================================
cat("[4/5] Generating figures...\n")

# ── F1: VLW stacked bar by income group ──────────────────────────────────
f1_data <- df_2023 %>%
  group_by(disease, LMIC_group) %>%
  summarise(VLW = sum(VLW), VLW_lower = sum(VLW_lower), VLW_upper = sum(VLW_upper),
            .groups = "drop") %>%
  mutate(disease = factor(disease, levels = disease_levels),
         LMIC_group = factor(LMIC_group, levels = income_levels))

p_f1 <- ggplot(f1_data, aes(x = disease, y = VLW, fill = LMIC_group)) +
  geom_col(colour = "grey30", linewidth = 0.2, position = "stack") +
  scale_fill_manual(values = income_colors, name = "Income Group") +
  scale_x_discrete(labels = c("Depressive disorders" = "Depressive\ndisorders",
                               "Anxiety disorders"    = "Anxiety\ndisorders")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.06)),
                     labels = function(x) paste0(round(x, 1), " bn USD")) +
  labs(x = "Disease", y = "VLW (billion USD, 2023)") +
  theme_nm() +
  theme(plot.title        = element_blank(),
        axis.line.x       = element_line(colour = "grey30", linewidth = 0.4),
        axis.line.y       = element_line(colour = "grey30", linewidth = 0.4),
        axis.ticks        = element_line(colour = "grey30", linewidth = 0.3),
        axis.ticks.length = unit(0.15, "cm"),
        axis.text.x       = element_text(size = 7.5, colour = "grey20", lineheight = 1.1),
        axis.text.y       = element_text(size = 7.5, colour = "grey20"),
        axis.title.x      = element_text(size = 8, face = "bold", colour = "grey20"),
        axis.title.y      = element_text(size = 8, face = "bold", colour = "grey20"),
        legend.position   = "top")

pdf(file.path(out_dir, "F1_VLW_disease_income_stacked.pdf"), width = 5.5, height = 4)
print(p_f1)
dev.off()
cat("   F1 saved.\n")

# ── F2: VLW/GDP boxplot + jitter ─────────────────────────────────────────
p_f2 <- ggplot(df_2023, aes(x = disease, y = VLW_GDP_pct, fill = disease)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.5, linewidth = 0.3) +
  geom_jitter(aes(colour = LMIC_group), width = 0.15, size = 0.9, alpha = 0.65) +
  scale_fill_manual(values = disease_colors, guide = "none") +
  scale_colour_manual(values = income_colors, name = "Income Group") +
  scale_y_continuous(labels = function(x) paste0(round(x, 2), "%")) +
  labs(x = NULL, y = "VLW / GDP (%)") +
  theme_nm() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1),
        legend.position = "top",
        axis.line  = element_line(colour = "grey30", linewidth = 0.4),
        axis.ticks = element_line(colour = "grey30", linewidth = 0.3))

pdf(file.path(out_dir, "F2_VLW_GDP_pct_boxplot.pdf"), width = 5, height = 4)
print(p_f2)
dev.off()
cat("   F2 saved.\n")

# ── F1_F2_combined_ab: F1 (left, a) | F2 (right, b) ──────────────────────
# Strip individual legends; collect into one shared legend at top
p_f1_ab <- p_f1 +
  labs(tag = "a", x = NULL) +
  theme(legend.position = "none",
        plot.margin     = margin(2, 8, 2, 4, "pt"))

p_f2_ab <- p_f2 +
  labs(tag = "b") +
  theme(legend.position = "none",
        plot.margin     = margin(2, 4, 2, 8, "pt"))

p_f1f2_ab <- (p_f1_ab | p_f2_ab) +
  plot_layout(widths = c(1, 1.3), guides = "collect") &
  theme(plot.tag         = element_text(size = 10, face = "bold", colour = "grey10"),
        legend.position  = "top",
        legend.key.size  = unit(0.35, "cm"),
        legend.title     = element_text(size = 7.5, face = "bold"),
        legend.text      = element_text(size = 7),
        legend.box.spacing = unit(2, "pt"))

pdf(file.path(out_dir, "F1_F2_combined_ab.pdf"), width = 9.5, height = 4)
print(p_f1f2_ab)
dev.off()
cat("   F1_F2_combined_ab saved.\n")

# ── F3a: Temporal trend — 2-panel VLW ────────────────────────────────────
trend_panel <- trend_by_disease %>%
  mutate(disease = factor(disease, levels = disease_levels))

p_f3 <- ggplot(trend_panel, aes(x = year, y = VLW_total)) +
  geom_hline(yintercept = -Inf, colour = "grey30", linewidth = 0.4) +
  geom_vline(xintercept = -Inf, colour = "grey30", linewidth = 0.4) +
  geom_ribbon(aes(ymin = VLW_lower, ymax = VLW_upper, fill = disease),
              alpha = 0.20, colour = NA) +
  geom_line(aes(colour = disease), linewidth = 0.8) +
  geom_vline(xintercept = 2019, colour = "grey50", linewidth = 0.4, linetype = "dashed") +
  scale_colour_manual(values = disease_colors, guide = "none") +
  scale_fill_manual(values   = disease_colors, guide = "none") +
  scale_x_continuous(breaks = c(1990, 2000, 2010, 2019, 2023),
                     labels = c("1990","2000","2010","2019","2023")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)),
                     labels = function(x) paste0(round(x, 2), " bn")) +
  facet_wrap(~ disease, scales = "free_y", ncol = 2) +
  labs(x = "Year", y = "Total VLW (billion USD)") +
  theme_nm(base_size = 8) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6.5),
        axis.ticks  = element_line(colour = "grey30", linewidth = 0.3),
        panel.border = element_blank(), axis.line = element_blank())

pdf(file.path(out_dir, "F3a_VLW_trend_2panel.pdf"), width = 7, height = 3.8)
print(p_f3)
dev.off()
cat("   F3a saved (2-panel VLW trend).\n")

# ── F3b: Temporal trend — VLW/GDP 2-panel ────────────────────────────────
trend_gdp_panel <- df_vlw %>%
  filter(sex_name == "Both") %>%
  group_by(year, disease) %>%
  summarise(
    VLW_GDP_wtd       = weighted.mean(VLW_GDP_pct,       GDP_PPP_total),
    VLW_GDP_wtd_lower = weighted.mean(VLW_GDP_pct_lower, GDP_PPP_total),
    VLW_GDP_wtd_upper = weighted.mean(VLW_GDP_pct_upper, GDP_PPP_total),
    .groups = "drop"
  ) %>%
  mutate(disease = factor(disease, levels = disease_levels))

p_f3_gdp <- ggplot(trend_gdp_panel, aes(x = year, y = VLW_GDP_wtd)) +
  geom_hline(yintercept = -Inf, colour = "grey30", linewidth = 0.4) +
  geom_vline(xintercept = -Inf, colour = "grey30", linewidth = 0.4) +
  geom_ribbon(aes(ymin = VLW_GDP_wtd_lower, ymax = VLW_GDP_wtd_upper, fill = disease),
              alpha = 0.20, colour = NA) +
  geom_line(aes(colour = disease), linewidth = 0.8) +
  geom_vline(xintercept = 2019, colour = "grey50", linewidth = 0.4, linetype = "dashed") +
  scale_colour_manual(values = disease_colors, guide = "none") +
  scale_fill_manual(values   = disease_colors, guide = "none") +
  scale_x_continuous(breaks = c(1990, 2000, 2010, 2019, 2023),
                     labels = c("1990","2000","2010","2019","2023")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)),
                     labels = function(x) paste0(round(x, 3), "%")) +
  facet_wrap(~ disease, scales = "free_y", ncol = 2) +
  labs(x = "Year", y = "GDP-weighted VLW / GDP (%)") +
  theme_nm(base_size = 8) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 6.5),
        axis.ticks  = element_line(colour = "grey30", linewidth = 0.3),
        panel.border = element_blank(), axis.line = element_blank())

pdf(file.path(out_dir, "F3b_VLW_GDP_trend_2panel.pdf"), width = 7, height = 3.8)
print(p_f3_gdp)
dev.off()
cat("   F3b saved (2-panel VLW/GDP trend).\n")

# ── F3_combined_ab: F3a (top, a) / F3b (bottom, b) ───────────────────────
p_f3_top <- p_f3 +
  labs(tag = "a", x = NULL) +
  theme(axis.text.x       = element_blank(),
        axis.ticks.x      = element_blank(),
        plot.margin       = margin(6, 6, 10, 6, "pt"))

p_f3_bot <- p_f3_gdp +
  labs(tag = "b") +
  theme(plot.margin = margin(10, 6, 6, 6, "pt"))

p_f3_ab <- (p_f3_top / p_f3_bot) +
  plot_layout(heights = c(1, 1)) &
  theme(plot.tag = element_text(size = 10, face = "bold", colour = "grey10"))

pdf(file.path(out_dir, "F3_combined_ab.pdf"), width = 7.5, height = 8)
print(p_f3_ab)
dev.off()
cat("   F3_combined_ab saved (a: VLW, b: VLW/GDP).\n")

# ── F3c: Overlay both diseases on one chart ──────────────────────────────
p_f3c <- ggplot(trend_panel, aes(x = year, y = VLW_total, colour = disease)) +
  geom_ribbon(aes(ymin = VLW_lower, ymax = VLW_upper, fill = disease),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = 2019, colour = "grey50", linewidth = 0.4, linetype = "dashed") +
  annotate("text", x = 2019.3, y = max(trend_panel$VLW_total) * 0.95,
           label = "COVID-19", size = 2.5, colour = "grey40", hjust = 0) +
  scale_colour_manual(values = disease_colors, name = "Disease") +
  scale_fill_manual(values   = disease_colors, guide = "none") +
  scale_x_continuous(breaks = seq(1990, 2023, 5)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.06)),
                     labels = function(x) paste0(round(x, 2), " bn")) +
  labs(x = "Year", y = "Total VLW (billion USD)") +
  theme_nm() +
  theme(legend.position = "top",
        axis.line  = element_line(colour = "grey30", linewidth = 0.4),
        axis.ticks = element_line(colour = "grey30", linewidth = 0.3))

pdf(file.path(out_dir, "F3c_VLW_trend_overlay.pdf"), width = 6, height = 3.8)
print(p_f3c)
dev.off()
cat("   F3c saved (overlay trend).\n")

# ── F4: All LMIC countries — sorted horizontal bar ───────────────────────
all_countries_ordered <- tbl_country_combined %>%
  arrange(VLW_comb_GDP_pct) %>%
  pull(location_name)

f4_data <- df_2023 %>%
  filter(location_name %in% all_countries_ordered) %>%
  mutate(location_name = factor(location_name, levels = all_countries_ordered),
         disease = factor(disease, levels = disease_levels))

n_ctry      <- length(all_countries_ordered)
fig_h_f4    <- max(10, n_ctry * 0.18)

income_label_colors <- tbl_country_combined %>%
  arrange(VLW_comb_GDP_pct) %>%
  mutate(label_col = income_colors[as.character(LMIC_group)]) %>%
  pull(label_col)

p_f4a <- ggplot(f4_data, aes(x = VLW_GDP_pct, y = location_name, fill = disease)) +
  geom_col(colour = NA, linewidth = 0, position = "stack") +
  scale_fill_manual(values = disease_colors, name = "Disease") +
  scale_x_continuous(labels = function(x) paste0(round(x, 2), "%"),
                     expand = expansion(mult = c(0, 0.04))) +
  labs(x = "VLW / GDP (%)", y = NULL, tag = "a") +
  theme_nm(base_size = 7) +
  theme(legend.position    = "none",
        axis.text.y        = element_text(size = 5, colour = income_label_colors),
        panel.grid.major.y = element_blank(),
        plot.tag           = element_text(size = 9, face = "bold"))

p_f4b <- ggplot(f4_data, aes(x = VLW, y = location_name, fill = disease)) +
  geom_col(colour = NA, linewidth = 0, position = "stack") +
  scale_fill_manual(values = disease_colors, name = "Disease") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.04))) +
  labs(x = "VLW (billion USD)", y = NULL, tag = "b") +
  theme_nm(base_size = 7) +
  theme(legend.position    = "bottom",
        axis.text.y        = element_blank(),
        axis.ticks.y       = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.tag           = element_text(size = 9, face = "bold"))

p_f4_comb <- (p_f4a | p_f4b) +
  plot_layout(widths = c(1.1, 1), guides = "collect") &
  theme(legend.position = "bottom",
        legend.key.size = unit(0.3, "cm"),
        legend.text     = element_text(size = 6.5),
        legend.title    = element_text(size = 7, face = "bold"))

pdf(file.path(out_dir, "F4_allLMIC_VLW_GDP_and_VLW_ab.pdf"), width = 11, height = fig_h_f4)
print(p_f4_comb)
dev.off()
cat("   F4 saved (", n_ctry, "countries).\n")

# ── F4c: Top 20 by VLW absolute ──────────────────────────────────────────
top20 <- df_2023 %>%
  group_by(location_id, location_name, LMIC_group) %>%
  summarise(VLW_total = sum(VLW), .groups = "drop") %>%
  arrange(desc(VLW_total)) %>% slice_head(n = 20) %>%
  arrange(VLW_total) %>% pull(location_name)

f4c_data <- df_2023 %>%
  filter(location_name %in% top20) %>%
  mutate(location_name = factor(location_name, levels = top20),
         disease = factor(disease, levels = disease_levels))

p_t20a <- ggplot(f4c_data, aes(x = VLW, y = location_name, fill = disease)) +
  geom_col(colour = NA, linewidth = 0, position = "stack") +
  scale_fill_manual(values = disease_colors, name = "Disease") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.04))) +
  labs(x = "VLW (billion USD)", y = NULL, tag = "a") +
  theme_nm(base_size = 8) +
  theme(legend.position = "none", axis.text.y = element_text(size = 7),
        panel.grid.major.y = element_blank(),
        plot.tag = element_text(size = 9, face = "bold"))

p_t20b <- ggplot(f4c_data, aes(x = VLW_GDP_pct, y = location_name, fill = disease)) +
  geom_col(colour = NA, linewidth = 0, position = "stack") +
  scale_fill_manual(values = disease_colors, name = "Disease") +
  scale_x_continuous(labels = function(x) paste0(round(x, 2), "%"),
                     expand = expansion(mult = c(0, 0.04))) +
  labs(x = "VLW / GDP (%)", y = NULL, tag = "b") +
  theme_nm(base_size = 8) +
  theme(legend.position = "bottom", axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), panel.grid.major.y = element_blank(),
        plot.tag = element_text(size = 9, face = "bold"))

p_top20 <- (p_t20a | p_t20b) +
  plot_layout(widths = c(1, 1), guides = "collect") &
  theme(legend.position = "bottom",
        legend.key.size = unit(0.35, "cm"),
        legend.text     = element_text(size = 7),
        legend.title    = element_text(size = 7.5, face = "bold"))

pdf(file.path(out_dir, "F4c_top20_VLW_absolute.pdf"), width = 9, height = 5.5)
print(p_top20)
dev.off()
cat("   F4c saved (top 20).\n")

# ── F5: Scatter — GDP pc vs combined VLW/GDP ─────────────────────────────
p_f5 <- ggplot(tbl_combined,
               aes(x = GDP_pc_PPP, y = VLW_comb_GDP_pct,
                   colour = LMIC_group, size = DALY_combined)) +
  geom_point(alpha = 0.65, stroke = 0.3) +
  scale_x_log10(labels = dollar_format(),
                breaks  = c(500, 1000, 2000, 5000, 10000, 25000)) +
  scale_y_continuous(labels = function(x) paste0(round(x, 3), "%")) +
  scale_colour_manual(values = income_colors, name = "Income Group") +
  scale_size_continuous(name   = "Total DALYs",
                        range  = c(1.5, 8),
                        breaks = c(1e4, 1e5, 5e5, 2e6),
                        labels = c("10K","100K","500K","2M")) +
  labs(x = "GDP per capita (PPP, USD)",
       y = "Combined VLW / GDP (%)") +
  theme_nm() +
  theme(legend.position = "right",
        axis.line  = element_line(colour = "grey30", linewidth = 0.4),
        axis.ticks = element_line(colour = "grey30", linewidth = 0.3))

pdf(file.path(out_dir, "F5_scatter_GDPpc_vs_combined_VLW.pdf"), width = 7, height = 4.5)
print(p_f5)
dev.off()
cat("   F5 saved.\n")

# ── F6: Choropleth maps ──────────────────────────────────────────────────
map_colors <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A",
                "#EF3B2C","#CB181D","#99000D")

sf::sf_use_s2(FALSE)
world_valid <- sf::st_make_valid(df_world)
world_crop  <- sf::st_crop(world_valid, xmin = -180, xmax = 180,
                                        ymin = -57,   ymax = 85)
sf::sf_use_s2(TRUE)

world_comb <- world_crop %>%
  left_join(tbl_combined %>% select(location_id, VLW_comb_GDP_pct, VLW_combined),
            by = "location_id")

p_map_gdp <- ggplot(world_comb) +
  geom_sf(aes(fill = VLW_comb_GDP_pct), colour = "grey40", linewidth = 0.06) +
  scale_fill_gradientn(
    colours = map_colors, na.value = "grey92", trans = "log1p",
    breaks  = c(0, 0.01, 0.05, 0.1, 0.2, 0.5, 1),
    labels  = c("0%","0.01%","0.05%","0.1%","0.2%","0.5%","1%"),
    name    = "Combined\nVLW/GDP (%)") +
  coord_sf(expand = FALSE) + theme_nm_map() +
  theme(legend.key.width = unit(0.35, "cm"), legend.key.height = unit(0.55, "cm"))

p_map_vlw <- ggplot(world_comb) +
  geom_sf(aes(fill = VLW_combined), colour = "grey40", linewidth = 0.06) +
  scale_fill_gradientn(
    colours = map_colors, na.value = "grey92", trans = "log1p",
    breaks  = c(0, 0.001, 0.01, 0.1, 0.5, 2, 10),
    labels  = c("0","0.001","0.01","0.1","0.5","2","10 bn"),
    name    = "Combined\nVLW (bn USD)") +
  coord_sf(expand = FALSE) + theme_nm_map() +
  theme(legend.key.width = unit(0.35, "cm"), legend.key.height = unit(0.55, "cm"))

p_map_ab <- (p_map_vlw + labs(tag = "a")) / (p_map_gdp + labs(tag = "b")) +
  plot_layout(ncol = 1) &
  theme(plot.tag = element_text(size = 9, face = "bold"))

pdf(file.path(out_dir, "F6a_map_combined_ab.pdf"), width = 8, height = 7.8)
print(p_map_ab)
dev.off()
cat("   F6a saved (combined map a/b).\n")

# Per-disease maps
for (d in disease_levels) {
  d_map_data <- df_2023 %>%
    filter(disease == d) %>%
    select(location_id, VLW, VLW_GDP_pct)

  world_d <- world_crop %>%
    left_join(d_map_data, by = "location_id")

  col_pal <- disease_palettes[[d]]

  p_d_gdp <- ggplot(world_d) +
    geom_sf(aes(fill = VLW_GDP_pct), colour = "grey40", linewidth = 0.06) +
    scale_fill_gradientn(colours = col_pal, na.value = "grey92", trans = "log1p",
                         name = "VLW/GDP (%)") +
    coord_sf(expand = FALSE) + theme_nm_map()

  p_d_vlw <- ggplot(world_d) +
    geom_sf(aes(fill = VLW), colour = "grey40", linewidth = 0.06) +
    scale_fill_gradientn(colours = col_pal, na.value = "grey92", trans = "log1p",
                         name = "VLW (bn USD)") +
    coord_sf(expand = FALSE) + theme_nm_map()

  p_d_ab <- (p_d_vlw + labs(tag = "a")) / (p_d_gdp + labs(tag = "b")) +
    plot_layout(ncol = 1) &
    theme(plot.tag = element_text(size = 9, face = "bold"))

  fname <- paste0("F6b_map_", gsub(" ", "_", d), "_ab.pdf")
  pdf(file.path(out_dir, fname), width = 8, height = 7.8)
  print(p_d_ab)
  dev.off()
}
cat("   F6b saved (per-disease maps).\n")

# ── F_age: Age-stratified figures (Both sexes, 2023) ─────────────────────
age_vlw_both <- df_age_2023 %>%
  filter(sex_name == "Both") %>%
  group_by(age_name, disease) %>%
  summarise(VLW_total  = sum(VLW),
            VLW_lower  = sum(VLW_lower),
            VLW_upper  = sum(VLW_upper),
            DALY_total = sum(DALY),
            .groups = "drop") %>%
  mutate(age_name = factor(age_name, levels = age_levels),
         disease  = factor(disease,  levels = disease_levels))

# ── Shared axis theme for bar-panel figures ───────────────────────────────
axis_theme_bar <- theme(
  axis.line.x       = element_line(colour = "grey30", linewidth = 0.4),
  axis.line.y       = element_line(colour = "grey30", linewidth = 0.4),
  axis.ticks        = element_line(colour = "grey30", linewidth = 0.3),
  axis.ticks.length = unit(0.15, "cm"),
  axis.text.x       = element_text(size = 7.5, colour = "grey20"),
  axis.text.y       = element_text(size = 7.5, colour = "grey20"),
  axis.title.x      = element_text(size = 8, face = "bold", colour = "grey20"),
  axis.title.y      = element_text(size = 8, face = "bold", colour = "grey20"),
  strip.text        = element_text(size = 8, face = "bold", colour = "grey10"),
  panel.border      = element_blank()
)

# F_age1: VLW by age band — 2-panel (one per disease)
p_age1 <- ggplot(age_vlw_both, aes(x = age_name, y = VLW_total, fill = disease)) +
  geom_col(colour = "grey30", linewidth = 0.2) +
  geom_errorbar(aes(ymin = VLW_lower, ymax = VLW_upper),
                width = 0.25, linewidth = 0.45, colour = "grey20") +
  scale_fill_manual(values = disease_colors, guide = "none") +
  scale_x_discrete(labels = age_labels) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.10)),
                     labels = function(x) paste0(round(x, 3), " bn")) +
  facet_wrap(~ disease, scales = "free_y", ncol = 2) +
  labs(x = "Age group (years)", y = "VLW (billion USD)") +
  theme_nm(base_size = 8) +
  axis_theme_bar

pdf(file.path(out_dir, "F_age1_VLW_2panel_both.pdf"), width = 7, height = 3.8)
print(p_age1)
dev.off()
cat("   F_age1 saved.\n")

# F_age2: VLW/GDP by age band
age_gdp_both <- df_age_2023 %>%
  filter(sex_name == "Both") %>%
  group_by(age_name, disease) %>%
  summarise(VLW_GDP_wtd       = weighted.mean(VLW_GDP_pct,       GDP_PPP_total),
            VLW_GDP_wtd_lower = weighted.mean(VLW_GDP_pct_lower, GDP_PPP_total),
            VLW_GDP_wtd_upper = weighted.mean(VLW_GDP_pct_upper, GDP_PPP_total),
            .groups = "drop") %>%
  mutate(age_name = factor(age_name, levels = age_levels),
         disease  = factor(disease,  levels = disease_levels))

p_age2 <- ggplot(age_gdp_both, aes(x = age_name, y = VLW_GDP_wtd, fill = disease)) +
  geom_col(colour = "grey30", linewidth = 0.2) +
  geom_errorbar(aes(ymin = VLW_GDP_wtd_lower, ymax = VLW_GDP_wtd_upper),
                width = 0.25, linewidth = 0.45, colour = "grey20") +
  scale_fill_manual(values = disease_colors, guide = "none") +
  scale_x_discrete(labels = age_labels) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.10)),
                     labels = function(x) paste0(formatC(x, format = "f", digits = 4), "%")) +
  facet_wrap(~ disease, scales = "free_y", ncol = 2) +
  labs(x = "Age group (years)", y = "GDP-weighted VLW / GDP (%)") +
  theme_nm(base_size = 8) +
  axis_theme_bar

pdf(file.path(out_dir, "F_age2_VLW_GDP_2panel_both.pdf"), width = 7, height = 3.8)
print(p_age2)
dev.off()
cat("   F_age2 saved.\n")

# ── F_age_combined_ab: F_age1 (top, a) / F_age2 (bottom, b) ─────────────
p_age_ab <- (p_age1 + labs(tag = "a")) / (p_age2 + labs(tag = "b")) +
  plot_layout(ncol = 1, heights = c(1, 1)) &
  theme(plot.tag = element_text(size = 10, face = "bold", colour = "grey10"),
        plot.margin = margin(4, 6, 4, 6, "pt"))

pdf(file.path(out_dir, "F_age_combined_ab.pdf"), width = 7, height = 7.8)
print(p_age_ab)
dev.off()
cat("   F_age_combined_ab saved (a: VLW, b: VLW/GDP).\n")

# F_age3: Male vs Female VLW by age band (2×2 grid)
age_sex_vlw <- df_age_2023 %>%
  filter(sex_name %in% c("Male", "Female")) %>%
  group_by(age_name, disease, sex_name) %>%
  summarise(VLW_total = sum(VLW), VLW_lower = sum(VLW_lower), VLW_upper = sum(VLW_upper),
            .groups = "drop") %>%
  mutate(age_name = factor(age_name, levels = age_levels),
         disease  = factor(disease,  levels = disease_levels),
         sex_name = factor(sex_name, levels = c("Male", "Female")))

p_age3 <- ggplot(age_sex_vlw, aes(x = age_name, y = VLW_total, fill = sex_name)) +
  geom_col(position = "dodge", colour = "grey30", linewidth = 0.2) +
  scale_fill_manual(values = c("Male" = "#2980B9", "Female" = "#E74C3C"), name = "Sex") +
  scale_x_discrete(labels = age_labels) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.10)),
                     labels = function(x) paste0(round(x, 3), " bn")) +
  facet_wrap(~ disease, scales = "free_y", ncol = 2) +
  labs(x = "Age group (years)", y = "VLW (billion USD)") +
  theme_nm(base_size = 8) +
  axis_theme_bar +
  theme(legend.position = "top",
        legend.key.size = unit(0.35, "cm"),
        legend.title    = element_text(size = 7.5, face = "bold"),
        legend.text     = element_text(size = 7))

pdf(file.path(out_dir, "F_age3_VLW_sex_comparison.pdf"), width = 7, height = 3.8)
print(p_age3)
dev.off()
cat("   F_age3 saved.\n")

# ── F_trend_income: VLW trend stratified by income group ─────────────────
trend_income_disease <- df_vlw %>%
  filter(sex_name == "Both") %>%
  group_by(year, LMIC_group, disease) %>%
  summarise(VLW_total = sum(VLW), VLW_lower = sum(VLW_lower), VLW_upper = sum(VLW_upper),
            .groups = "drop") %>%
  mutate(disease    = factor(disease,    levels = disease_levels),
         LMIC_group = factor(LMIC_group, levels = income_levels))

p_trend_inc <- ggplot(trend_income_disease,
                      aes(x = year, y = VLW_total, colour = LMIC_group)) +
  geom_hline(yintercept = -Inf, colour = "grey30", linewidth = 0.4) +
  geom_vline(xintercept = -Inf, colour = "grey30", linewidth = 0.4) +
  geom_ribbon(aes(ymin = VLW_lower, ymax = VLW_upper, fill = LMIC_group),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_vline(xintercept = 2019, colour = "grey50", linewidth = 0.45, linetype = "dashed") +
  annotate("text", x = 2019.4, y = -Inf, vjust = -0.5,
           label = "2019", size = 2.2, colour = "grey45", hjust = 0) +
  scale_colour_manual(values = income_colors, name = "Income Group") +
  scale_fill_manual(values   = income_colors, guide = "none") +
  scale_x_continuous(breaks = c(1990, 2000, 2010, 2019, 2023),
                     labels = c("1990","2000","2010","2019","2023"),
                     expand = expansion(mult = c(0.01, 0.03))) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.10)),
                     labels = function(x) paste0(round(x, 2), " bn")) +
  facet_wrap(~ disease, scales = "free_y", ncol = 2) +
  labs(x = "Year", y = "Total VLW (billion USD)") +
  theme_nm(base_size = 8) +
  theme(legend.position  = "top",
        legend.key.size  = unit(0.35, "cm"),
        legend.title     = element_text(size = 7.5, face = "bold"),
        legend.text      = element_text(size = 7),
        axis.text.x      = element_text(angle = 30, hjust = 1, size = 7, colour = "grey20"),
        axis.text.y      = element_text(size = 7, colour = "grey20"),
        axis.title.x     = element_text(size = 8, face = "bold", colour = "grey20"),
        axis.title.y     = element_text(size = 8, face = "bold", colour = "grey20"),
        axis.ticks       = element_line(colour = "grey30", linewidth = 0.3),
        axis.ticks.length = unit(0.15, "cm"),
        strip.text       = element_text(size = 8, face = "bold", colour = "grey10"),
        panel.border     = element_blank(),
        axis.line        = element_blank())

pdf(file.path(out_dir, "F_trend_VLW_by_income_2panel.pdf"), width = 7.5, height = 4)
print(p_trend_inc)
dev.off()
cat("   F_trend_income saved.\n")

# ===========================================================================
# 5. SUMMARY PRINT
# ===========================================================================
cat("\n[5/5] Summary (Both sexes, <20 years, all LMICs, 2023):\n")
cat(sprintf("   %-30s  %s\n", "Disease", "VLW (bn USD) | VLW/GDP (%)"))
cat(sprintf("   %-30s  %s\n", "-------", "----------------------------"))

sum_2023 <- df_2023 %>%
  group_by(disease) %>%
  summarise(VLW_total = sum(VLW), DALY_total = sum(DALY),
            VLW_GDP_wtd = weighted.mean(VLW_GDP_pct, GDP_PPP_total),
            .groups = "drop")

for (i in seq_len(nrow(sum_2023))) {
  cat(sprintf("   %-30s  %.4f bn USD | %.4f%%\n",
              sum_2023$disease[i], sum_2023$VLW_total[i], sum_2023$VLW_GDP_wtd[i]))
}
cat(sprintf("\n   %-30s  %.4f bn USD | %.4f%%\n", "COMBINED (2 diseases)",
            sum(sum_2023$VLW_total),
            weighted.mean(tbl_combined$VLW_comb_GDP_pct, tbl_combined$GDP_PPP_total)))

cat("\n══════════════════════════════════════════════════════════════════\n")
cat("  All outputs saved to:", out_dir, "\n")
cat("══════════════════════════════════════════════════════════════════\n")

# ============================================================
# EdNet 데이터 전처리
# ============================================================
library(data.table)
library(tools)

# -------------------------
# 0) 경로 설정
# -------------------------
P <- list(
  KT1="C:/data/KT1", KT2="C:/data/KT2", KT4="C:/data/KT4",
  Q ="C:/data/contents/questions.csv",
  
  KT4_pay="C:/data/KT4_pay",
  KT1_pay="C:/data/KT1_pay",
  KT2_pay="C:/data/KT2_pay",
  KT1_cor="C:/data/KT1_pay_cor",
  KT2_res="C:/data/KT2_pay_res",
  
  pay_users_pay="C:/data/pay_users_pay.csv",
  pay_users_all="C:/data/pay_users_all.csv",
  
  item_out="C:/data/analysis_table_item.csv"
)

mk <- function(x) if (!dir.exists(x)) dir.create(x, recursive=TRUE)
uid_list <- function(dir) file_path_sans_ext(list.files(dir, "\\.csv$", full.names=FALSE))
clean_abcd <- function(x){
  x <- tolower(trimws(as.character(x)))
  x[!x %chin% c("a","b","c","d")] <- NA_character_
  x
}

# -------------------------
# 1) KT4: 유료 사용자
# -------------------------
mk(P$KT4_pay)
kt4_files <- list.files(P$KT4, "\\.csv$", full.names=TRUE)

pay_users <- character()
for (f in kt4_files) {
  dt <- fread(f, select="action_type", showProgress=FALSE)
  if (any(dt$action_type == "pay", na.rm=TRUE)) {
    pay_users <- c(pay_users, file_path_sans_ext(basename(f)))
  }
}
pay_users <- unique(pay_users)
fwrite(data.table(user_id=pay_users), P$pay_users_pay)

file.copy(file.path(P$KT4, paste0(pay_users, ".csv")),
          file.path(P$KT4_pay, paste0(pay_users, ".csv")),
          overwrite=TRUE)

# -------------------------
# 2) 교집합 코호트
# -------------------------
mk(P$KT1_pay); mk(P$KT2_pay)

u4 <- uid_list(P$KT4_pay)
u1 <- uid_list(P$KT1)
u2 <- uid_list(P$KT2)

pay_all <- Reduce(intersect, list(u4,u1,u2))
fwrite(data.table(user_id=pay_all), P$pay_users_all)

file.copy(file.path(P$KT1, paste0(pay_all, ".csv")),
          file.path(P$KT1_pay, paste0(pay_all, ".csv")), overwrite=TRUE)
file.copy(file.path(P$KT2, paste0(pay_all, ".csv")),
          file.path(P$KT2_pay, paste0(pay_all, ".csv")), overwrite=TRUE)

# -------------------------
# 3) questions 로드
# -------------------------
q <- fread(P$Q, showProgress=FALSE)
q[, question_id := trimws(as.character(question_id))]
q[, correct_answer := clean_abcd(correct_answer)]
q_key <- q[, .(question_id, correct_answer, part, tags, bundle_id)]
setkey(q_key, question_id)

# -------------------------
# 4) KT1: correct 생성
# -------------------------
mk(P$KT1_cor)
kt1p <- list.files(P$KT1_pay, "\\.csv$", full.names=TRUE)

for (f in kt1p) {
  uid <- file_path_sans_ext(basename(f))
  dt <- fread(f, select=c("question_id","user_answer","elapsed_time"), showProgress=FALSE)
  dt[, question_id := trimws(as.character(question_id))]
  dt[, user_answer := clean_abcd(user_answer)]
  dt <- q_key[dt, on="question_id"]
  dt[, correct := fifelse(!is.na(user_answer) & !is.na(correct_answer),
                          as.integer(user_answer == correct_answer), NA_integer_)]
  fwrite(dt[, .(question_id, user_answer, elapsed_time, correct)],
         file.path(P$KT1_cor, paste0(uid, ".csv")))
}

# -------------------------
# 5) KT2: response_change 생성
# -------------------------
mk(P$KT2_res)
kt2p <- list.files(P$KT2_pay, "\\.csv$", full.names=TRUE)

for (f in kt2p) {
  uid <- file_path_sans_ext(basename(f))
  dt <- fread(f, select=c("timestamp","action_type","item_id","user_answer"), showProgress=FALSE)
  
  dt <- dt[action_type %chin% c("respond","submit")]
  setorder(dt, timestamp)
  
  dt[, submit_idx := cumsum(action_type == "submit")]
  dt[action_type=="respond", submit_idx := submit_idx + 1L]
  
  max_submit <- dt[action_type=="submit", max(submit_idx, na.rm=TRUE)]
  if (!is.finite(max_submit)) max_submit <- 0L
  
  r <- dt[action_type=="respond" & submit_idx>=1 & submit_idx<=max_submit]
  r <- r[grepl("^q", item_id)]
  r[, question_id := item_id]
  r[, user_answer := clean_abcd(user_answer)]
  r <- r[!is.na(user_answer)]
  
  if (nrow(r)==0) {
    fwrite(data.table(submit_idx=integer(), question_id=character(), response_change=integer()),
           file.path(P$KT2_res, paste0(uid, ".csv")))
    next
  }
  
  setorder(r, submit_idx, question_id, timestamp)
  r[, prev := shift(user_answer), by=.(submit_idx, question_id)]
  r[, chg := as.integer(!is.na(prev) & user_answer != prev)]
  
  out <- r[, .(response_change = sum(chg, na.rm=TRUE)), by=.(submit_idx, question_id)]
  fwrite(out, file.path(P$KT2_res, paste0(uid, ".csv")))
}

# -------------------------
# 6) 문항 단위 집계/병합
# -------------------------
# KT1 집계
kt1_cor_files <- list.files(P$KT1_cor, "\\.csv$", full.names=TRUE)
kt1_all <- rbindlist(lapply(kt1_cor_files, function(f){
  uid <- file_path_sans_ext(basename(f))
  x <- fread(f, showProgress=FALSE)
  x[, user_id := uid]
  x
}), use.names=TRUE, fill=TRUE)

kt1_all <- kt1_all[!is.na(correct)]
kt1_item <- kt1_all[, .(
  n_students_KT1 = uniqueN(user_id),
  n_attempts_KT1 = .N,
  correct_rate   = mean(correct),
  mean_time      = mean(elapsed_time, na.rm=TRUE)
), by=question_id]

# KT2 집계
kt2_res_files <- list.files(P$KT2_res, "\\.csv$", full.names=TRUE)
kt2_all <- rbindlist(lapply(kt2_res_files, function(f){
  uid <- file_path_sans_ext(basename(f))
  x <- fread(f, showProgress=FALSE)
  x[, user_id := uid]
  x
}), use.names=TRUE, fill=TRUE)

kt2_item <- kt2_all[, .(
  n_students_KT2 = uniqueN(user_id),
  n_attempts_KT2 = .N,
  mean_change    = mean(response_change, na.rm=TRUE),
  change_rate    = mean(response_change >= 1, na.rm=TRUE),
  mean_change_cond = {
    y <- response_change[response_change >= 1]
    if (length(y)==0) NA_real_ else mean(y)
  }
), by=question_id]

# 병합 + 메타
item <- merge(q_key[, .(question_id, part, tags, bundle_id)],
              kt1_item, by="question_id", all.x=TRUE)
item <- merge(item, kt2_item, by="question_id", all.x=TRUE)
item[, n_students := fifelse(!is.na(n_students_KT1), n_students_KT1, n_students_KT2)]

fwrite(item, P$item_out)


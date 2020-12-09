
library(tidyverse)
library(fst)

# Load data
df = read.fst("../Data/32interval30_CE_11hours_limit.fst")


mean(df$n_ping)
sd(df$n_ping)

# Add SCELag

## select unique drivers
x <- unique(df$driver)

dl <- data.frame()
for (i in 1:length(x)) { 
        f <- filter(df, driver == x[i])
        for (k in 1:nrow(f)) {
                temp <- (f$start_time[k] - (24 * 3600 *7) <= f$start_time) & 
                        (f$start_time[k] > f$start_time)
                f$nCE_hist[k] <- sum(f$nCE[temp])
                f$drive_time_week[k] <- sum(f$interval_time[temp])
                if (f$drive_time_week[k] != 0) {
                        f$rt_ratio[k] <- (f$nCE_hist[k]/f$drive_time_week[k]) 
                }
                else {
                        f$rt_ratio[k] <- 0  
                }
        }
        dl <- rbind(dl,f)
        print(i)
}

# Compute SCE rate
dd = dl %>%
        mutate(SCE = ifelse(nCE > 0, 1, 0)) %>%
        group_by(driver) %>%
        mutate(total_SCE = sum(nCE), total_mile = sum(distance),
               total_time = sum(interval_time), 
               SCE_rate_mile = total_SCE/total_mile,
               SCE_rate_time = total_SCE/total_time)

d = unique(dd[,c(1,38:42)])
d[1:497,] %>%
        ggplot() +
        geom_point(aes(x = driver, y = SCE_rate_mile))

dv_outlier = d[which.max(d$SCE_rate_mile),]

# Remove the driver with the higher SCE rate

dd = dd %>%
        filter(SCE_rate_mile < dv_outlier$SCE_rate_mile[1])

dd = dd[,-c(2:4,6,8:12,17,23:26,28,30,34:35,38:42)]

write.fst(dd, "../Data/REG500_SCELag7.fst")
write.csv(dd, "../Data/REG500_SCELag7_date.csv")



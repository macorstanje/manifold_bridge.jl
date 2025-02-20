setwd("~/.julia/dev/manifold_bridge.jl/output")
library(tidyverse)
library(ggforce)

mytheme = theme_bw()
theme_set(mytheme)

####################### plot some guided paths

# drift V1(x) = -20x
# drift V2(x) = x
# drift V3(x) = [5 *(1-x[1]^2-x[2]^2)^2, 0]

someguidedpaths <- read_csv("someguidedpaths_minus20.csv", col_names = FALSE) %>% rename(time=X1, x=X2, y=X3, id=X4) %>% 
  filter(id <5)
             
path1 <- someguidedpaths %>% filter(id==1) 
n <- nrow(path1)
x0 = path1[1,2:3]
xT = path1[n,2:3]

someguidedpaths1 <- read_csv("someguidedpaths_1.csv", col_names = FALSE) %>% rename(time=X1, x=X2, y=X3, id=X4) %>%   filter(id <5)

someguidedpathsright <- read_csv("someguidedpaths_right.csv", col_names = FALSE) %>% rename(time=X1, x=X2, y=X3, id=X4) %>%   filter(id <5)

d <- bind_rows(someguidedpaths, someguidedpaths1, someguidedpathsright) %>% 
  mutate(a=rep(c("V1","V2", "V3"),each=4*n) )

p <- d %>% ggplot() + geom_path(aes(x=x,y=y, colour=time)) +  
  geom_circle(aes(x0 = 0, y0 = 0, r = 1), color = "black") +
  annotate("point", x = as.numeric(x0[1]), y = as.numeric(x0[2]), color = "red") + 
  annotate("point", x = as.numeric(xT[1]), y = as.numeric(xT[2]), color = "blue") +
  facet_grid(a ~ id) + coord_fixed(ratio=1) +labs(x="", y="")  +
  scale_y_continuous(breaks=c(-1,0,1)) +
  scale_x_continuous(breaks=c(-1,0,1)) +
  theme(strip.text.x = element_blank(), strip.background.x = element_blank()) +
  theme(strip.text.y = element_text(angle = 0))
#  theme(legend.position = "bottom")
p

ggsave("hyperbolic_guidedpaths.pdf",width=7,height=4)


####################### with V3 a difficult case 
diffpaths  <- read_csv("someguidedpaths_right_difficult.csv", col_names = FALSE) %>% rename(time=X1, x=X2, y=X3, id=X4) %>% 
  filter(id <5)

path1 <- diffpaths %>% filter(id==1) 
n <- nrow(path1)
x0 = path1[1,2:3]
xT = path1[n,2:3]

pdiff <- diffpaths %>% filter(id <3) %>% ggplot() + geom_path(aes(x=x,y=y, colour=time)) +  
  geom_circle(aes(x0 = 0, y0 = 0, r = 1), color = "black") +
  annotate("point", x = as.numeric(x0[1]), y = as.numeric(x0[2]), color = "red") + 
  annotate("point", x = as.numeric(xT[1]), y = as.numeric(xT[2]), color = "blue") +
  facet_wrap( ~ id) + coord_fixed(ratio=1) +labs(x="", y="")  +
  scale_y_continuous(breaks=c(-1,0,1)) +
  scale_x_continuous(breaks=c(-1,0,1)) + 
  theme(strip.background = element_blank(), strip.text = element_blank())
pdiff

#ggsave("hyperbolic_guidedpaths_V3b.pdf",width=5,height=4)

####################### plot pCN results 



mcmcpaths <- read_csv("mcmcpaths.csv", col_names = FALSE) %>% rename(time=X1, x=X2, y=X3, iteration=X4)

path1 <- mcmcpaths %>% filter(iteration==1) 
x0 = path1[1,2:3]
xT = path1[nrow(path1),2:3]


# d <-  mcmcpaths %>% filter(iteration %in% c(1,5,10))
# 
# ggplot() + geom_path(data =d, aes(x=x,y=y, group=iteration, colour=time)) +  
#   geom_circle(aes(x0 = 0, y0 = 0, r = 1), color = "black") +
#      annotate("point", x = as.numeric(x0[1]), y = as.numeric(x0[2]), color = "red")+ 
#      annotate("point", x = as.numeric(xT[1]), y = as.numeric(xT[2]), color = "blue") +
#      facet_wrap(~iteration) +   coord_fixed(ratio=1)+labs(x="", y="")

#%>% filter(iteration==2)
M = max(mcmcpaths$iteration)
mcmcpaths   %>% filter(iteration %in% seq(1,M,by=50)) %>%
  ggplot(aes(x=x,y=y,group=iteration,colour=iteration)) + geom_path()    +  
  scale_color_viridis_c(option = "viridis") +
  #scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0)+
  geom_circle(aes(x0 = 0, y0 = 0, r = 1), color = "black") +
  annotate("point", x = as.numeric(x0[1]), y = as.numeric(x0[2]), color = "red")+ 
  annotate("point", x = as.numeric(xT[1]), y = as.numeric(xT[2]), color = "blue") +
   coord_fixed(ratio=1)+labs(x="", y="")

ggsave("hyperbolic_pcn.pdf",width=5,height=4)


################## visualise vector field 
library(ggplot2)
library(dplyr)

# Define grid inside unit disk
grid_points <- expand.grid(x1 = seq(-0.95, 0.95, length.out = 8), 
                           x2 = seq(-0.95, 0.95, length.out = 8)) %>%
  filter(x1^2 + x2^2 < 1)  # Keep only points inside the unit disk

scaling <- 0.1
# Compute the vector field
vector_field <- grid_points %>%
  mutate(V1 = 5.0 * (1 - x1^2 - x2^2)^2,  # First component of V(x)
         V2 = 0.0,                        # Second component of V(x)
       norm = 1,#sqrt(V1^2 + V2^2),        # Normalize for consistent arrow sizes
         V1 = V1 / norm * scaling,            # Scale arrows for visibility
         V2 = V2 / norm * scaling)

# Plot the vector field on the Poincaré disk
pV3 <- ggplot() +
  geom_segment(data = vector_field, 
               aes(x = x1, y = x2, xend = x1 + V1, yend = x2 + V2), 
               arrow = arrow(length = unit(scaling, "cm")), 
               color = "blue") +
  geom_circle(aes(x0 = 0, y0 = 0, r = 1), color = "black") +  # Poincaré disk boundary
  coord_fixed(ratio = 1) +labs(x="", y="")
pV3

# +
#   theme_minimal() +
#   labs(x = expression(x[1]), y = expression(x[2]))


library(patchwork)
grid.arrange(pV3, pdiff,nrow=1)
combined_plot <- pV3 + pdiff + plot_layout(widths = c(1, 2))

# Print the combined plot
combined_plot

ggsave("specialplotsV3.pdf",width=7,height=3)

#lang racket
(define (sum_product lst) (list (apply + lst) (apply * lst)))

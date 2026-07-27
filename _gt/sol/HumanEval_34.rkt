#lang racket
(define (unique lst) (sort (remove-duplicates lst) <))

#lang racket
(define (total_match l1 l2)
  (define (tc l) (apply + (map string-length l)))
  (if (<= (tc l1) (tc l2)) l1 l2))

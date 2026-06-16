#lang racket
(define (compare game guess)
  (map (lambda (g h) (abs (- g h))) game guess))

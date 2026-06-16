#lang racket
(define (simplify x n)
  (define (parse s) (map string->number (string-split s "/")))
  (define xs (parse x)) (define ns (parse n))
  (= 0 (modulo (* (first xs) (first ns)) (* (second xs) (second ns)))))

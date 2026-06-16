#lang racket
(define (parse_music s)
  (map (lambda (tok)
         (cond [(string=? tok "o") 4]
               [(string=? tok "o|") 2]
               [(string=? tok ".|") 1]
               [else 0]))
       (string-split s)))
